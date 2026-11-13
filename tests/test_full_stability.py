"""Full-strategy protocol and independent anchor regression checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.compare_lfba_auto_vote import parse_options
from lfba_comparison.runner import configuration, frozen_protocol
from lfba_comparison.stability_config import stability_protocol
from lfba_comparison.stability import _read_protocol, run_stability
from lfba_comparison.common import write_json

class FullStabilityTests(unittest.TestCase):
    def test_shared_split_retains_fixed_anchor_for_all_arms(self):
        opts=parse_options(['--stability','--smoke'])
        for method in ('capture','auto_vote','lfba_original'):
            args=configuration(opts,100,method,Path('unused'))
            self.assertTrue(args.retain_anchor)
            self.assertEqual(args.anchor_idx,23470)
        protocol=frozen_protocol(opts)
        self.assertEqual(protocol['anchor'],23470)
        self.assertEqual(protocol['methods'],['auto_vote','lfba_original'])
        self.assertEqual(protocol['version'],5)

    def test_fixed_anchor_retained_in_bounded_shared_split(self):
        from dataset.experiment import build_datasets,_bounded_ids,split_training_ids
        from tests.helpers import TinyData
        from utils.config import parse_args
        base=TinyData(1000)
        selected=_bounded_ids(base.targets.numpy(),500,100)
        missing=next(i for i in range(1000) if i not in selected)
        for anchor in (missing,split_training_ids(base.targets.numpy(),.1,100)[1][0]):
            args=parse_args(['--dataset','CIFAR10','--attack','LFBA','--anchor_idx',str(anchor),
                             '--seed','100','--train_limit','500','--eval_limit','100'])
            args.retain_anchor=True
            with patch('dataset.experiment.CIFAR10_VFL',side_effect=lambda *a,**kw:TinyData(1000)):
                train,val,test,state=build_datasets(args)
                train2,val2,_,_=build_datasets(args,state)
            self.assertIn(anchor,train.ids)
            self.assertNotIn(anchor,val.ids)
            self.assertEqual([len(train),len(val),len(test)],[450,50,100])
            self.assertEqual((train.ids,val.ids),(train2.ids,val2.ids))

    def test_old_protocol_reportable_but_not_resumable_as_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            opts=parse_options(['--stability','--run','--resume','--output',tmp])
            opts.anchor_source='auto_vote'
            legacy=stability_protocol(opts)
            self.assertEqual(legacy['version'],1)
            write_json(Path(tmp)/'protocol.json',legacy)
            self.assertEqual(_read_protocol(Path(tmp)),legacy)
            opts.anchor_source='fixed'
            with patch('lfba_comparison.stability.run_experiment',side_effect=AssertionError('training forbidden')):
                with self.assertRaisesRegex(ValueError,'identical full stability matrix'):
                    run_stability(opts)

    def test_fixed_pair_allows_distinct_targets_and_counts(self):
        from lfba_comparison.records import verified_pair
        def group(path,method,*args,**kwargs):
            fixed=method=='lfba_original'
            return dict(anchor=23470 if fixed else 7,target_class=1 if fixed else 2,
                initial_digest='i',data_digest='d',first_prefix={'epoch':1},start_state='s',
                test=dict(counts={'acc':100,'asr':91 if fixed else 89})),{},[]
        def config(path):
            return dict(poison_rate=.1,select_rate=.5,pretrain_stage=1 if path.parent.name=='lfba_original' else 3)
        with patch('lfba_comparison.records.verified_group',side_effect=group), patch('lfba_comparison.records.read_json',side_effect=config):
            result=verified_pair(Path('seed_100'),40,100,.1,.5,anchor_source='fixed')
            self.assertNotEqual(result['auto_vote']['anchor'],result['lfba_original']['anchor'])
            with self.assertRaisesRegex(ValueError,'fixed anchor'):
                verified_pair(Path('seed_100'),40,100,.1,.5,anchor_source='fixed',fixed_anchor=99)

if __name__=='__main__':unittest.main()
