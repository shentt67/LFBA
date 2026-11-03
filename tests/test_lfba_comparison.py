"""Synthetic policies and state only: no forward, backward, train or model test."""
import contextlib
import hashlib
import importlib.util
import io
import logging
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from attack.attack import get_near_index
from attack.neighbor_vote import select_all
from lfba_comparison.common import digest, read_json, write_json
from lfba_comparison.policy import ANCHOR, METHODS, fixed_choice, purity, schedule
from lfba_comparison.report import summarize, write_report
from lfba_comparison.runner import export, verify_checkpoint
from lfba_comparison.trainer import ComparisonTrainer

ROOT=Path(__file__).resolve().parents[1]


def observations():
    g=torch.randn(40,7,generator=torch.Generator().manual_seed(41))
    ids=torch.tensor([ANCHOR]+list(range(39)))
    return [dict(epoch=i+1,ids=ids.clone(),gradients=g.clone()) for i in range(3)]


class View:
    ids=[ANCHOR]+list(range(39))
    targets=np.arange(ANCHOR+1)%10
    def __len__(self):return len(self.ids)
    def set_attack(self,plan,args,dimensions,labels=None):
        self.plan=dict(plan);self.labels=dict(labels or {})


def trainer(directory,method,checkpoint=None):
    args=SimpleNamespace(seed=106,epoch=20,lr=.001,comparison_method=method,
        anchor_idx=ANCHOR,target_label=0,block_size=128,results_dir=str(directory))
    loader=SimpleNamespace(dataset=View(),generator=torch.Generator().manual_seed(7))
    obj=ComparisonTrainer(torch.device('cpu'),[],[],None,[],None,loader,None,None,[],
        logging.getLogger('comparison-test'),args,checkpoint,data_state=dict(train_ids=View.ids))
    obj.indexes_epoch=observations()[0]['ids'];obj.grad_vec_epoch=observations()[0]['gradients']
    obj.observations=observations()[:1 if method!='auto_vote' else 3]
    return obj


class PolicyTests(unittest.TestCase):
    def test_exact_formal_budgets_and_epochs(self):
        raw=schedule('lfba_original',20,4500);matched=schedule('lfba_matched',20,4500);vote=schedule('auto_vote',20,4500)
        self.assertEqual(raw,[0]+[2250]*19)
        self.assertEqual(matched,[0]+[2014]*3+[2013]*16)
        self.assertEqual(vote,[0]*3+[2250]*17)
        self.assertEqual([sum(raw),sum(matched),sum(vote)],[42750,38250,38250])

    def test_smoke_schedule_generalization(self):
        self.assertEqual(schedule('lfba_original',5,18),[0,9,9,9,9])
        self.assertEqual(schedule('lfba_matched',5,18),[0,5,5,4,4])
        self.assertEqual(schedule('auto_vote',5,18),[0,0,0,9,9])

    def test_baseline_is_identical_to_existing_legacy_sort_and_only_first_round(self):
        obs=observations();ids=obs[0]['ids'];g=obs[0]['gradients']
        expected=sorted(ids[get_near_index(g[0],g,4,'legacy_l1',ids)].tolist())
        for method in METHODS[:2]:self.assertEqual(fixed_choice(obs,method,4)['candidates'],expected)
        obs[1]['gradients'].fill_(float('nan'));obs[2]['gradients'].fill_(0)
        self.assertEqual(fixed_choice(obs,'lfba_original',4)['candidates'],expected)

    def test_fixed_selection_is_id_aligned(self):
        obs=observations();before=fixed_choice(obs,'lfba_original',4)
        order=torch.arange(39,-1,-1)
        obs[0]['ids']=obs[0]['ids'][order];obs[0]['gradients']=obs[0]['gradients'][order]
        self.assertEqual(before,fixed_choice(obs,'lfba_original',4))

    def test_automatic_selector_is_byte_identical_to_completed_experiment(self):
        actual=hashlib.sha256((ROOT/'attack'/'neighbor_vote.py').read_bytes()).hexdigest()
        self.assertEqual(actual,'ee720d602bcedebca7090acbe05f964079f180bb0c090573db4545a9e2f93ad8')

    def test_purity_reports_both_denominators(self):
        labels=np.array([1,1,0,0])
        result=purity(labels,[0,1,2],0)
        self.assertEqual(result['full'],2/3);self.assertEqual(result['without_anchor'],.5)
        self.assertEqual(result['without_anchor_denominator'],2)
        self.assertFalse(purity(labels,[1,2],0)['anchor_in_pool'])


class StateTests(unittest.TestCase):
    def test_start_epochs_freeze_and_global_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in METHODS:
                output=Path(directory)/method;output.mkdir();obj=trainer(output,method)
                window=3 if method=='auto_vote' else 1
                for ep in range(window):obj._configure_attack(ep)
                self.assertEqual(obj.cumulative,0);self.assertIsNone(obj.choice)
                before=random.getstate();obj._configure_attack(window)
                self.assertEqual(before,random.getstate())
                self.assertEqual(obj.plans[0]['epoch'],window+1)
                self.assertEqual(obj.label_overrides,{})
                candidates=list(obj.choice['candidates'])
                obj._configure_attack(window+1)
                self.assertEqual(candidates,obj.choice['candidates'])
                obj.poison_indexes=[]
                with self.assertRaises(ValueError):obj._configure_attack(window+2)

    def test_baselines_share_candidates_and_vote_uses_existing_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            choices=[]
            for method in METHODS:
                output=Path(directory)/method;output.mkdir();obj=trainer(output,method)
                obj._configure_attack(3 if method=='auto_vote' else 1);choices.append(obj.choice)
            self.assertEqual(choices[0]['candidates'],choices[1]['candidates'])
            self.assertEqual(choices[2],select_all(observations(),4)['results']['auto_vote'])

    def test_resume_restores_policy_history_budget_and_donor_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            obj=trainer(directory,'lfba_original');obj._configure_attack(1);obj.completed_epoch=2
            obj.audit=[{'epoch':1},{'epoch':2}];obj.prefix=[{'epoch':1}]
            with patch('utils.trainer.capture_rng',return_value={'fixed':1}):snapshot=obj._snapshot()
            other=trainer(directory,'lfba_original',snapshot)
            with patch('utils.trainer.restore_rng'):other._restore()
            self.assertEqual(digest(obj.observations),digest(other.observations))
            self.assertEqual(obj.audit,other.audit);self.assertEqual(obj.cumulative,other.cumulative)
            obj._configure_attack(2);other._configure_attack(2)
            self.assertEqual(obj.current_plan,other.current_plan)
            self.assertEqual(obj.donor_rng.getstate(),other.donor_rng.getstate())

    def test_adam_restore_does_not_mutate_shared_prefix_reference(self):
        # Adam load_state_dict can retain CPU step tensors by reference. Simulate
        # that state mutation directly; never call forward, backward or step.
        with tempfile.TemporaryDirectory() as directory:
            parameter=torch.nn.Parameter(torch.tensor([1.]))
            optimizer=torch.optim.Adam([parameter],lr=.001)
            optimizer.state[parameter]=dict(step=torch.tensor(2.),exp_avg=torch.zeros_like(parameter),
                                             exp_avg_sq=torch.zeros_like(parameter))
            capture=trainer(directory,'capture');capture.optimizer_list=[optimizer];capture.completed_epoch=1
            with patch('utils.trainer.capture_rng',return_value={'fixed':1}):prefix=capture._snapshot()
            before=digest(prefix)
            recorded=[]
            for method in ('lfba_original','lfba_matched','auto_vote'):
                other=trainer(directory,method,prefix)
                other_parameter=torch.nn.Parameter(torch.tensor([1.]))
                other_optimizer=torch.optim.Adam([other_parameter],lr=.001)
                other.optimizer_list=[other_optimizer]
                with patch('utils.trainer.restore_rng'),patch('lfba_comparison.trainer.capture_rng',return_value={'fixed':1}):
                    other._restore()
                recorded.append(other.start_state)
                loaded_step=other_optimizer.state[other_parameter]['step']
                self.assertNotEqual(loaded_step.data_ptr(),prefix['optimizer'][0]['state'][0]['step'].data_ptr())
                loaded_step.add_(8)
                other_optimizer.state[other_parameter]['exp_avg'].add_(3)
                self.assertEqual(digest(prefix),before)
                self.assertEqual(prefix['optimizer'][0]['state'][0]['step'].item(),2.)
            self.assertEqual(len(set(recorded)),1)

    def test_shared_prefix_load_checks_rng_and_rejects_cross_method(self):
        with tempfile.TemporaryDirectory() as directory:
            capture=trainer(directory,'capture');capture.completed_epoch=1
            with patch('utils.trainer.capture_rng',return_value={'fixed':1}):snapshot=capture._snapshot()
            for method in METHODS:
                other=trainer(directory,method,snapshot)
                with patch('utils.trainer.restore_rng'),patch('lfba_comparison.trainer.capture_rng',return_value={'fixed':1}):
                    other._restore()
                self.assertIsNotNone(other.start_state);self.assertEqual(other.elapsed_prior,0)
            other=trainer(directory,'auto_vote',snapshot)
            with patch('utils.trainer.restore_rng'),patch('lfba_comparison.trainer.capture_rng',return_value={'wrong':1}):
                with self.assertRaises(ValueError):other._restore()
            snapshot['comparison']['method']='lfba_original'
            with patch('utils.trainer.restore_rng'):
                with self.assertRaises(ValueError):trainer(directory,'auto_vote',snapshot)._restore()

    def test_durable_json_keeps_prior_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json';write_json(path,{'ok':1})
            with self.assertRaises(ValueError):write_json(path,{'bad':float('nan')})
            self.assertEqual(read_json(path),{'ok':1})

    def test_final_checkpoint_verifier_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            capture=trainer(directory,'capture');capture.completed_epoch=1
            capture.prefix=[dict(epoch=1,model='shared')]
            capture.audit=[dict(epoch=1,recipient_count=0,cumulative_replacements=0,candidate_count=0)]
            with patch('utils.trainer.capture_rng',return_value={'fixed':1}):prefix=capture._snapshot()
            for method in METHODS:
                output=Path(directory)/method;output.mkdir()
                obj=trainer(output,method,prefix)
                with patch('utils.trainer.restore_rng'),patch('lfba_comparison.trainer.capture_rng',return_value={'fixed':1}):obj._restore()
                for ep in range(1,20):
                    obj._configure_attack(ep)
                    if method=='auto_vote' and ep<3:obj.observations.append(observations()[ep])
                    candidates=[] if obj.choice is None else obj.choice['candidates']
                    obj.audit.append(dict(epoch=ep+1,recipient_count=len(obj.current_plan),
                        cumulative_replacements=obj.cumulative,candidate_count=len(candidates),
                        candidate_digest=digest(candidates)))
                obj.completed_epoch=20
                with patch('utils.trainer.capture_rng',return_value={'fixed':1}):final=obj._snapshot()
                verify_checkpoint(final,method,20,prefix)
                final['comparison']['audit'][1]['recipient_count']+=1
                with self.assertRaises(ValueError):verify_checkpoint(final,method,20,prefix)


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('comparison_cli',ROOT/'scripts'/'compare_lfba_auto_vote.py')
        cls.cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.cli)

    def test_default_and_dry_run_never_dispatch_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out'
            with patch('lfba_comparison.runner.run_experiment',side_effect=AssertionError('training forbidden')):
                with contextlib.redirect_stdout(io.StringIO()):
                    for flags in ([],['--dry-run']):self.cli.main([*flags,'--output',str(out),'--data-dir',str(Path(directory)/'input')])
            self.assertFalse(out.exists())

    def test_explicit_run_dispatch_is_mocked_without_training(self):
        with patch.object(self.cli,'dry_run',return_value=0),patch('lfba_comparison.runner.run_experiment') as execute:
            self.cli.main(['--run']);execute.assert_called_once()

    def test_report_only_cannot_start_training_or_inference(self):
        with patch('lfba_comparison.report.write_report') as report,patch('lfba_comparison.runner.run_experiment',side_effect=AssertionError()):
            self.cli.main(['--report-only']);report.assert_called_once()

    def test_conflicting_and_invalid_options_fail(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for args in (['--resume'],['--run','--dry-run'],['--seeds','106','106'],['--block-size','0']):
                with self.assertRaises(SystemExit):self.cli.parse_options(args)

    def test_default_data_directory_in_main_repository_and_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            repository=Path(directory)/'repository'
            for code_root in (repository,repository/'.worktrees'/'comparison'):
                with patch.object(self.cli,'ROOT',code_root):
                    options=self.cli.parse_options([])
                self.assertEqual(options.data_dir,(repository/'data').resolve())


class ReportTests(unittest.TestCase):
    def records(self,seed=106):
        rows=[]
        for method in METHODS:
            choice=dict(method=method,anchor_id=ANCHOR,candidates=[1,2,3,4])
            new=method=='auto_vote'
            rows.append(dict(seed=seed,method=method,epoch=5,anchor=ANCHOR,target_class=0,
                test=dict(acc=.75,asr=.80+.03*new,poison_acc=.2),
                candidate_purity=dict(full=.75,full_denominator=4,without_anchor=.75,without_anchor_denominator=4),
                recipient_purity=.7,candidate_digest=digest(choice['candidates']),
                initial_digest='init',data_digest='data',first_prefix={'epoch':1},start_state='start',
                cumulative_replacements=sum(schedule(method,5,4)),final_model_digest='model-'+method,
                timing={'shared_first_round':1},peak_host_memory_bytes=1,peak_cuda_memory_bytes=0))
        return rows

    def test_thresholds_and_shared_candidates(self):
        rows=sum([self.records(seed) for seed in (106,107,108)],[])
        result=summarize(rows)
        self.assertTrue(result['lfba_original']['meets_threshold'])
        self.assertTrue(result['lfba_matched']['meets_threshold'])
        rows[1]['candidate_digest']='different'
        with self.assertRaises(ValueError):summarize(rows)

    def test_report_uses_only_synthetic_saved_records(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_json(root/'protocol.json',dict(seeds=[106],epochs=5,smoke=True))
            for record in self.records():
                method=record['method'];folder=root/'seed_106'/method;folder.mkdir(parents=True)
                choice=dict(method=method,anchor_id=ANCHOR,candidates=[1,2,3,4])
                write_json(folder/'choice.json',choice);write_json(folder/'metrics.json',record)
                write_json(folder/'independent_evaluation.json',dict(epoch=5,test=record['test'],final_model_digest=record['final_model_digest']))
                expected=schedule(method,5,4);clean=3 if method=='auto_vote' else 1
                rows=[dict(epoch=i+1,recipient_count=n,cumulative_replacements=sum(expected[:i+1]),
                    candidate_count=0 if i<clean else 4,candidate_digest=digest([] if i<clean else choice['candidates'])) for i,n in enumerate(expected)]
                with (folder/'epoch_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                write_json(folder/'history.json',[dict(epoch=i+1,validation=dict(acc=.75,asr=None if i<clean else .8)) for i in range(5)])
            with patch('utils.trainer.Trainer.train',side_effect=AssertionError('training forbidden')), \
                 patch('utils.trainer.Trainer.test',side_effect=AssertionError('inference forbidden')):
                write_report(root)
            self.assertTrue((root/'validation_curves_lfba_vs_vote.png').exists())
            self.assertFalse(read_json(root/'summary.json')['formal_verdict_applicable'])
            self.assertTrue((root/'budget_check.csv').exists())


if __name__=='__main__':unittest.main()
