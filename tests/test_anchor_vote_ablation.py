"""Anchor x neighbor voting: policy, provenance and reporting contracts."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lfba_comparison.anchor_vote_policy import select_fixed, METHODS


def observations():
    rng = torch.Generator().manual_seed(27)
    return [dict(epoch=i+1, ids=torch.tensor([23470]+list(range(19))),
                 gradients=torch.randn(20, 5, generator=rng)) for i in range(3)]


class FixedSelectionTests(unittest.TestCase):
    def test_exact_last_and_vote_rules(self):
        from attack.neighbor_vote import align_observations, ordered_candidates
        obs = observations(); selection = select_fixed(obs, 4)
        ids, gradients = align_observations(obs, 4)
        ids = ids.numpy(); position = int(np.flatnonzero(ids==23470)[0])
        scores = np.stack([(g@g[position]).numpy() for g in gradients])
        votes = np.zeros(len(ids), dtype=int)
        for row in scores:
            votes[ordered_candidates(ids, row, 4, 23470)] += 1
        for method, values, tally in [('fixed_single', scores[-1], None),
                                      ('fixed_vote', scores.mean(0), votes)]:
            choice = selection['results'][method]
            expected = sorted(ids[ordered_candidates(ids, values, 4, 23470, tally)].tolist())
            self.assertEqual(choice['candidates'], expected)
            self.assertEqual(choice['anchor_id'], 23470)
            self.assertNotIn(23470, expected)
        self.assertNotEqual(selection['results']['fixed_single']['candidates'],
                            selection['results']['fixed_vote']['candidates'])

    def test_common_valid_ids_permutation_and_invalid_anchor(self):
        obs = observations(); obs[0]['gradients'][1].zero_()
        first = select_fixed(obs, 4)
        for item in obs:
            item['ids'] = item['ids'].flip(0); item['gradients'] = item['gradients'].flip(0)
        self.assertEqual(first['results'], select_fixed(obs, 4)['results'])
        for result in first['results'].values(): self.assertNotIn(0, result['candidates'])
        obs = observations(); obs[1]['gradients'][0].zero_()
        with self.assertRaisesRegex(ValueError, 'anchor'): select_fixed(obs, 4)
        obs = observations(); obs[0]['ids'][0] = 777
        with self.assertRaises(ValueError): select_fixed(obs, 4)

    def test_vote_ties_and_epoch_validation(self):
        obs = observations()
        for item in obs: item['gradients'].fill_(1.)
        selected = select_fixed(obs, 4)
        self.assertEqual(selected['results']['fixed_vote']['candidates'], [0,1,2,3])
        # ID ties select the same four neighbors every round: only those have 3 votes.
        self.assertEqual(selected['results']['fixed_vote']['diagnostics']['boundary_tied_count'], 4)
        self.assertEqual(selected['results']['fixed_vote']['diagnostics']['selected_by_tiebreak'], 0)
        obs[0]['epoch'] = 2
        with self.assertRaises(ValueError): select_fixed(obs, 4)


class InterfaceTests(unittest.TestCase):
    def test_protocol_and_source_code_changes_are_rejected(self):
        from lfba_comparison.anchor_vote_runner import check_compatibility, require_identical_protocol, file_hash, ROOT
        protocol=dict(kind='core_ablation',dataset='CIFAR10',smoke=False,epochs=40,
                      batch_size=128,optimizer='Adam',lr=.001,lr_decay=False,early_stop=False,
                      poison_rate=.1,select_rate=.5,clean_rounds=3,attack_start=4,
                      retained_split_anchor=23470,final_model='last',similarity='cosine',
                      source_hashes={'lfba_comparison/core_trainer.py':file_hash(ROOT/'lfba_comparison/core_trainer.py')})
        check_compatibility(protocol)
        changed=copy.deepcopy(protocol);changed['epochs']=20
        with self.assertRaises(ValueError): check_compatibility(changed)
        changed=copy.deepcopy(protocol);changed['source_hashes']['lfba_comparison/core_trainer.py']='wrong'
        with self.assertRaises(ValueError): check_compatibility(changed)
        with self.assertRaises(ValueError): require_identical_protocol(protocol,changed)

    def test_no_training_without_explicit_run(self):
        from scripts.run_anchor_vote_ablation import parse_options, main
        args = parse_options([])
        self.assertEqual(args.seeds, [2004,592,3383]); self.assertFalse(args.run)
        with patch('lfba_comparison.anchor_vote_runner.preflight', return_value={'ready':True}), \
             patch('lfba_comparison.anchor_vote_runner.run') as run:
            self.assertEqual(main([]), 0); run.assert_not_called()

    def test_manifest_detects_file_changes_and_copy_preserves_source(self):
        from lfba_comparison.anchor_vote_runner import file_hash, verify_manifest, copy_prefix
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source=root/'source'; target=root/'new'/'prefix'
            source.write_bytes(b'checkpoint')
            manifest={str(source):file_hash(source)}
            copy_prefix(source, target, manifest[str(source)])
            self.assertEqual(source.read_bytes(), target.read_bytes())
            verify_manifest(manifest)
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError): verify_manifest(manifest)
            with self.assertRaises(ValueError): copy_prefix(source, target, file_hash(source))

    def test_report_arithmetic_missing_and_identity(self):
        from lfba_comparison.anchor_vote_report import summarize
        rows=[]
        for method, asr in [('fixed_single',.3),('fixed_vote',.5),
                            ('auto_rank_last',.6),('auto_vote',.9)]:
            rows.append(dict(seed=2004,method=method,anchor=1 if method.startswith('fixed') else 2,
                             target_class=0, start_state='same',data_digest='same',
                             test={'asr':asr,'acc':.8}))
        result=summarize(rows,[2004,592])
        self.assertAlmostEqual(result['interaction']['asr']['mean'], .1)
        self.assertAlmostEqual(result['comparisons']['auto_rank_last - fixed_single']['asr']['mean'], .3)
        self.assertEqual(result['interaction']['asr']['n'],1)
        self.assertFalse(result['all_complete'])
        bad=copy.deepcopy(rows);bad[-1]['start_state']='different'
        with self.assertRaises(ValueError): summarize(bad,[2004])

    def test_report_only_never_trains_and_marks_missing(self):
        from scripts.run_anchor_vote_ablation import main
        from lfba_comparison.common import write_json, read_json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_json(root/'protocol.json',dict(kind='anchor_vote_ablation',seeds=[2004],smoke=True,
                                                epochs=5,source_manifest=None))
            with patch('lfba_comparison.anchor_vote_report.load_records',return_value=([],[],[],{})), \
                 patch('lfba_comparison.anchor_vote_runner.run') as run:
                self.assertEqual(main(['--report-only','--output',str(root)]),1)
                run.assert_not_called()
            self.assertEqual(read_json(root/'summary.json')['new_complete'],0)


class StateTests(unittest.TestCase):
    def test_smoke_orchestration_report_and_resume_with_synthetic_cpu_models(self):
        from tests.helpers import make_trainer, TinyData
        from torch.utils.data import DataLoader
        from dataset.experiment import DataView
        from lfba_comparison.core_trainer import CoreTrainer
        from lfba_comparison.anchor_vote_runner import run
        from lfba_comparison.anchor_vote_report import write_report
        from lfba_comparison.common import read_json
        from scripts.run_anchor_vote_ablation import parse_options
        from utils.utils import set_seed

        def assemble(options,seed,method,output,checkpoint=None,choice=None):
            set_seed(seed)
            base=make_trainer(output,attack='LFBA',epoch=3 if method=='capture' else 5,
                              seed=seed,poison_rate=.1,early_stop=0,pretrain_stage=3,comparison_method=method)
            ids=list(range(449))+[23470]
            loader=DataLoader(DataView(TinyData(23471),ids),batch_size=128,shuffle=True,
                              generator=torch.Generator().manual_seed(seed+1))
            obj=CoreTrainer(base.device,base.model_list,[],None,base.optimizer_list,base.criterion,
                            loader,base.test_loader,base.test_asr_loader,base.trigger_dimensions,
                            base.logger,base.args,checkpoint,base.val_loader,base.val_asr_loader,
                            data_state={'train_ids':ids},choice=choice)
            return obj,None

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            args=parse_options(['--smoke','--output',str(Path(directory)/'batch'),'--device','-1'])
            with patch('lfba_comparison.anchor_vote_runner.assemble',side_effect=assemble), \
                 patch('lfba_comparison.core_runner.assemble',side_effect=assemble), \
                 patch('lfba_comparison.anchor_vote_runner.close'), \
                 patch('lfba_comparison.core_runner.close'):
                self.assertEqual(run(args),0)
                report=read_json(args.output/'summary.json')
                self.assertTrue(report['all_complete'])
                self.assertEqual(report['new_complete'],2)
                self.assertTrue(all(r['cumulative_replacements']==44 for r in report['records']))
                self.assertTrue((args.output/'validation_curves.png').is_file())
                self.assertTrue(write_report(args.output)['all_complete'])
                args.resume=True
                with patch('lfba_comparison.core_trainer.CoreTrainer.train',side_effect=AssertionError('must skip completed')):
                    self.assertEqual(run(args),0)

    def test_fixed_branches_restore_resume_budget_and_metadata(self):
        from tests.helpers import make_trainer
        from lfba_comparison.core_trainer import CoreTrainer
        from lfba_comparison.core_runner import verify_final, verify_result_metadata
        from lfba_comparison.core_policy import schedule
        from lfba_comparison.common import digest
        from utils.utils import set_seed
        from utils.checkpoint import load_checkpoint, atomic_save
        from utils.artifacts import runtime_checkpoint

        def make(path,method,checkpoint=None,choice=None):
            base=make_trainer(path,attack='LFBA',epoch=3 if method=='capture' else 5,
                              poison_rate=.1,early_stop=0,pretrain_stage=3,comparison_method=method)
            return CoreTrainer(base.device,base.model_list,[],None,base.optimizer_list,base.criterion,
                               base.train_loader,base.test_loader,base.test_asr_loader,base.trigger_dimensions,
                               base.logger,base.args,checkpoint,base.val_loader,base.val_asr_loader,
                               data_state={'train_ids':list(range(20))},choice=choice)

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);set_seed(2004)
            clean=make(root/'clean','capture');clean.train()
            prefix=load_checkpoint(runtime_checkpoint(root/'clean'));before=digest(prefix)
            choices=select_fixed(prefix['comparison']['observations'],2,anchor=1)['results']
            starts=[]
            for method in METHODS:
                choice=choices[method]
                full=make(root/method,method,prefix,choice);full.train()
                final=load_checkpoint(runtime_checkpoint(root/method))
                verify_final(final,prefix,choice,5)
                starts.append(full.start_state)
                self.assertEqual(full.cumulative,2)
                self.assertEqual([p['epoch'] for p in full.plans],[4,5])
                self.assertEqual(before,digest(prefix))
                part=make(root/'partial',method,prefix,choice)
                original=part._snapshot
                def interrupted():
                    snap=original()
                    if part.completed_epoch==4:
                        atomic_save(snap,runtime_checkpoint(root/'partial'))
                        raise InterruptedError('simulated')
                    return snap
                part._snapshot=interrupted
                with self.assertRaises(InterruptedError): part.train()
                resumed=make(root/'resume',method,load_checkpoint(runtime_checkpoint(root/'partial')),choice)
                resumed.train()
                self.assertEqual(full.plans,resumed.plans)
                self.assertEqual(digest([m.state_dict() for m in full.model_list]),
                                 digest([m.state_dict() for m in resumed.model_list]))
                self.assertEqual(full.test(4),resumed.test(4))
                wrong=copy.deepcopy(final);wrong['comparison']['plans'][0]['donors'][0]=choice['candidates'][0]
                with self.assertRaises(ValueError): verify_final(wrong,prefix,choice,5)
            self.assertEqual(len(set(starts)),1)
            self.assertEqual(sum(schedule(40,4500)),83250)
            self.assertEqual(sum(schedule(5,45)),44)


if __name__=='__main__': unittest.main()
