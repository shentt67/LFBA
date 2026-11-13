"""Small CPU checks for ordered same-anchor trials and failure recovery."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lfba_comparison.common import read_json
from lfba_comparison.records import verified_pair
from lfba_comparison.runner import run_experiment
from scripts.compare_lfba_auto_vote import parse_options
from tests.test_compact_comparison import tiny_assemble


def paired_assemble(options,seed,method,output,checkpoint=None,anchor=None):
    obj,handler=tiny_assemble(options,seed,method,output,checkpoint,anchor)
    obj.args.pretrain_stage=3 if method=='auto_vote' else 1
    obj.args.poison_rate=options.poison_rate
    obj.args.select_rate=options.select_rate
    obj.args.retain_anchor=False
    if anchor is not None:obj.args.anchor_idx=anchor
    return obj,handler


class StabilityPairingTests(unittest.TestCase):
    def test_insufficient_valid_gradients_are_not_silently_dropped(self):
        import torch
        from attack.neighbor_vote import align_observations
        gradients=torch.zeros(10,3);gradients[:4]=1
        with self.assertRaisesRegex(ValueError,'2P\\+1'):
            align_observations([dict(ids=torch.arange(10),gradients=gradients)],2)

    def options(self,root,seeds=('100',)):
        opts=parse_options(['--run','--smoke','--device','-1','--poison-rate','.2',
                            '--seeds',*seeds,'--output',str(root)])
        opts.stability=True;opts.anchor_source='auto_vote'
        opts.continue_on_error=True;opts.defer_cleanup=True
        return opts

    def test_order_anchor_and_pair_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            def assemble(*args,**kwargs):
                if args[2]!='capture' and args[4] is not None and args[4]['epoch']==1:
                    calls.append((args[2],args[5] if len(args)>5 else None))
                return paired_assemble(*args,**kwargs)
            with patch('lfba_comparison.runner.assemble',side_effect=assemble):
                run_experiment(self.options(root))
            pair=verified_pair(root/'seed_100',5,100,.2,.5)
            self.assertEqual([c[0] for c in calls],['auto_vote','lfba_original'])
            self.assertEqual(calls[1][1],pair['auto_vote']['anchor'])
            self.assertTrue(list(root.rglob('*.pth.tar')),'defer cleanup until global plots succeed')

    def test_failed_seed_does_not_block_other_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def assemble(*args,**kwargs):
                if args[1]==100 and args[2]=='auto_vote':raise RuntimeError('one trial failed')
                return paired_assemble(*args,**kwargs)
            with patch('lfba_comparison.runner.assemble',side_effect=assemble):
                result=run_experiment(self.options(root,('100','101')))
            self.assertEqual(result['failures'][0]['seed'],100)
            self.assertTrue((root/'seed_101/lfba_original/metrics.json').exists())
            self.assertFalse((root/'seed_100/lfba_original/metrics.json').exists())

    def test_no_fixed_anchor_in_stability_partition(self):
        from dataset.experiment import build_datasets,split_training_ids
        from tests.helpers import TinyData
        from utils.config import parse_args
        args=parse_args(['--dataset','CIFAR10','--attack','LFBA','--anchor_idx','99',
                         '--seed','100','--poison_rate','.1'])
        args.retain_anchor=False
        expected=split_training_ids(TinyData(100).targets.numpy(),.1,100,None)
        args.anchor_idx=expected[1][0]
        with patch('dataset.experiment.CIFAR10_VFL',side_effect=lambda *a,**kw:TinyData(100)):
            train,val,_,_=build_datasets(args)
        self.assertEqual((train.ids,val.ids),expected)


if __name__=='__main__':unittest.main()
