"""CPU integration checks for compact output, final reports and recovery."""
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.compare_lfba_auto_vote import parse_options
from lfba_comparison.common import read_json
from lfba_comparison.runner import run_experiment
from lfba_comparison.trainer import ComparisonTrainer
from tests.helpers import make_trainer
from utils.utils import set_seed


def tiny_assemble(options,seed,method,output,checkpoint=None,anchor=None):
    set_seed(seed)
    torch.set_num_threads(1)
    base=make_trainer(output,epoch=1 if method=='capture' else 5,attack='LFBA',
        seed=seed,early_stop=0,comparison_method=method,block_size=128)
    obj=ComparisonTrainer(base.device,base.model_list,[],None,base.optimizer_list,
        base.criterion,base.train_loader,base.test_loader,base.test_asr_loader,
        base.trigger_dimensions,base.logger,base.args,checkpoint,
        base.val_loader,base.val_asr_loader,dict(train_ids=list(range(20))))
    handler=logging.FileHandler(Path(output)/'experiment.log',encoding='utf-8')
    obj.logger.addHandler(handler)
    return obj,handler


class CompactComparisonTests(unittest.TestCase):
    def options(self,root):
        return parse_options(['--run','--smoke','--device','-1','--seeds','100',
            '--poison-rate','.2','--output',str(root)])

    def test_success_and_resume_without_checkpoint_or_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'batch';options=self.options(root)
            with patch('lfba_comparison.runner.assemble',side_effect=tiny_assemble):
                run_experiment(options)
            self.assertFalse(list(root.rglob('*.pth.tar')))
            self.assertFalse(list(root.rglob('selection.pt')))
            self.assertTrue((root/'results.md').exists())
            self.assertTrue((root/'validation_curves_lfba_vs_vote.png').exists())
            for method in ('lfba_original','auto_vote'):
                folder=root/'seed_100'/method
                self.assertEqual({p.name for p in folder.iterdir()},
                    {'config.json','experiment.log','experiment_data.json','metrics.json'})
                self.assertEqual(read_json(folder/'metrics.json')['status'],'complete')
            options.resume=True
            with patch('lfba_comparison.runner.assemble',side_effect=AssertionError('no retraining')):
                run_experiment(options)

    def test_plot_failure_keeps_runtime_until_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'batch';options=self.options(root)
            with patch('lfba_comparison.runner.assemble',side_effect=tiny_assemble), \
                 patch('lfba_comparison.report.write_report',side_effect=RuntimeError('plot failed')):
                with self.assertRaisesRegex(RuntimeError,'plot failed'):run_experiment(options)
            self.assertTrue(list(root.rglob('.runtime/last_checkpoint.pth.tar')))
            from scripts.compare_lfba_auto_vote import main as cli_main
            with patch('lfba_comparison.runner.assemble',side_effect=AssertionError('no retraining')):
                cli_main(['--report-only','--output',str(root)])
            self.assertFalse(list(root.rglob('*.pth.tar')))

    def test_failed_independent_evaluation_keeps_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'batch';options=self.options(root)
            original=ComparisonTrainer.test
            calls=0
            def fail(obj,ep=0,split='test'):
                nonlocal calls
                if split=='test':
                    calls+=1
                    if calls==2:raise RuntimeError('evaluation failed')
                return original(obj,ep,split)
            with patch('lfba_comparison.runner.assemble',side_effect=tiny_assemble), \
                 patch.object(ComparisonTrainer,'test',fail):
                with self.assertRaisesRegex(RuntimeError,'evaluation failed'):run_experiment(options)
            self.assertTrue(list(root.rglob('.runtime/last_checkpoint.pth.tar')))
            self.assertFalse((root/'seed_100'/'lfba_original'/'metrics.json').exists())

    def test_interrupted_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as directory:
            full=Path(directory)/'full';part=Path(directory)/'part'
            with patch('lfba_comparison.runner.assemble',side_effect=tiny_assemble):
                run_experiment(self.options(full))
            original=ComparisonTrainer._configure_attack
            def interrupt(obj,ep):
                if obj.method=='lfba_original' and ep==2:
                    raise RuntimeError('interrupted')
                return original(obj,ep)
            with patch('lfba_comparison.runner.assemble',side_effect=tiny_assemble), \
                 patch.object(ComparisonTrainer,'_configure_attack',interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    run_experiment(self.options(part))
            self.assertTrue((part/'seed_100/lfba_original/.runtime/last_checkpoint.pth.tar').exists())
            options=self.options(part);options.resume=True
            with patch('lfba_comparison.runner.assemble',side_effect=tiny_assemble):
                run_experiment(options)
            for method in ('lfba_original','auto_vote'):
                left=read_json(full/'seed_100'/method/'metrics.json')
                right=read_json(part/'seed_100'/method/'metrics.json')
                self.assertEqual(left['test'],right['test'])
                self.assertEqual(left['final_model_digest'],right['final_model_digest'])
                a=read_json(full/'seed_100'/method/'experiment_data.json')
                b=read_json(part/'seed_100'/method/'experiment_data.json')
                for key in ('history','choice','plans'):self.assertEqual(a[key],b[key])

    def test_metrics_write_failure_keeps_checkpoint(self):
        from lfba_comparison.common import write_json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'batch'
            def fail(path,value):
                if Path(path).name=='metrics.json':raise OSError('disk write failed')
                return write_json(path,value)
            with patch('lfba_comparison.runner.assemble',side_effect=tiny_assemble), \
                 patch('lfba_comparison.runner.write_json',side_effect=fail):
                with self.assertRaisesRegex(OSError,'disk write failed'):
                    run_experiment(self.options(root))
            self.assertTrue(list(root.rglob('.runtime/last_checkpoint.pth.tar')))

    def test_evaluator_assembly_preserves_bound_configuration(self):
        from lfba_comparison.runner import assemble,close
        from lfba_comparison.common import write_json
        from tests.test_entry import small_data
        from tests.helpers import Top,Bottom
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);options=self.options(output)
            config=dict(anchor_idx=19,target_label=1,data_state={'saved':True})
            write_json(output/'config.json',config)
            with patch('lfba_comparison.runner.build_datasets',side_effect=small_data), \
                 patch('lfba_comparison.runner.build_models',return_value=[Top(),Bottom(),Bottom()]), \
                 patch('lfba_comparison.runner.triggered_view',side_effect=lambda view,*args:view):
                obj,handler=assemble(options,100,'auto_vote',output,anchor=1)
                close(obj,handler)
            self.assertEqual(read_json(output/'config.json'),config)

    def test_cleanup_refuses_missing_configuration_or_corrupt_donor_records(self):
        from lfba_comparison.runner import cleanup_completed_batch
        from lfba_comparison.common import write_json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'batch';options=self.options(root)
            with patch('lfba_comparison.runner.assemble',side_effect=tiny_assemble):
                run_experiment(options)
            folder=root/'seed_100/lfba_original'
            recovery=folder/'.runtime/last_checkpoint.pth.tar'
            recovery.parent.mkdir();recovery.write_bytes(b'recovery must survive')
            config=read_json(folder/'config.json');(folder/'config.json').unlink()
            with self.assertRaises(ValueError):cleanup_completed_batch(root)
            self.assertTrue(recovery.exists())
            write_json(folder/'config.json',config)
            data=read_json(folder/'experiment_data.json')
            data['plans'][0]['donors'][0]=-1
            write_json(folder/'experiment_data.json',data)
            with self.assertRaises(ValueError):cleanup_completed_batch(root)
            self.assertTrue(recovery.exists())


if __name__=='__main__':unittest.main()
