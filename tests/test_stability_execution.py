"""End-to-end stability orchestration using tiny CPU training fixtures."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lfba_comparison.common import read_json
from lfba_comparison.stability import run_stability,report_stability
from scripts.compare_lfba_auto_vote import parse_options
from tests.test_stability_pairing import paired_assemble


class StabilityExecutionTests(unittest.TestCase):
    def setUp(self):
        # Tiny fixtures use sample 1 as the predeclared baseline anchor.
        from lfba_comparison.policy import ANCHORS
        patcher=patch.dict(ANCHORS,{'CIFAR10':1})
        patcher.start();self.addCleanup(patcher.stop)

    def options(self,output):
        return parse_options(['--stability','--run','--smoke','--device','-1',
            '--poison-rates','.2','--select-rates','.5','--seeds','100','--output',str(output)])

    def test_matrix_success_model_free_resume_and_protocol_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch';options=self.options(root)
            with patch('lfba_comparison.runner.assemble',side_effect=paired_assemble):
                self.assertEqual(run_stability(options),0)
            summary=read_json(root/'summary.json')
            self.assertEqual((summary['completed_pairs'],summary['completed_methods']),(2,4))
            self.assertFalse(list(root.rglob('*.pth.tar')))
            self.assertTrue((root/'stability_parameters.png').exists())
            self.assertIn('分配置结果说明',(root/'results.md').read_text(encoding='utf-8'))
            self.assertIn('参数变化趋势',(root/'results.md').read_text(encoding='utf-8'))
            for case in summary['cases']:
                original=read_json(root/case['id']/'seed_100/lfba_original/metrics.json')
                self.assertEqual(original['anchor'],1)
            trials=[read_json(root/c['id']/'seed_100/auto_vote/metrics.json')
                    for c in read_json(root/'protocol.json')['cases']]
            for key in ('initial_digest','data_digest','first_prefix','start_state'):
                self.assertEqual(trials[0][key],trials[1][key])
            options.resume=True
            with patch('lfba_comparison.runner.assemble',side_effect=AssertionError('no models')):
                self.assertEqual(run_stability(options),0)
                self.assertEqual(report_stability(root),0)
                options.seeds=[100,101]
                with self.assertRaisesRegex(ValueError,'identical full stability matrix'):
                    run_stability(options)

    def test_reduce_seeds_resumes_without_retraining_and_excludes_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch';options=self.options(root);options.seeds=[100,101]
            with patch('lfba_comparison.runner.assemble',side_effect=paired_assemble):
                self.assertEqual(run_stability(options),0)
            extra=root/'p_0p1_s_0p5/seed_101/auto_vote/metrics.json'
            original=extra.read_bytes()
            options.seeds=[100];options.resume=True;options.reduce_seeds=True
            with patch('lfba_comparison.runner.assemble',side_effect=AssertionError('no retraining')):
                self.assertEqual(run_stability(options),0)
            summary=read_json(root/'summary.json')
            self.assertEqual((summary['expected_pairs'],summary['completed_pairs']),(2,2))
            self.assertEqual(extra.read_bytes(),original)
            self.assertEqual(summary['seeds'],[100])

    def test_interrupted_full_strategy_matches_continuous(self):
        from lfba_comparison.trainer import ComparisonTrainer
        with tempfile.TemporaryDirectory() as tmp:
            full=Path(tmp)/'full';part=Path(tmp)/'part'
            with patch('lfba_comparison.runner.assemble',side_effect=paired_assemble):
                self.assertEqual(run_stability(self.options(full)),0)
            original=ComparisonTrainer._configure_attack
            def interrupt(obj,ep):
                if obj.method=='auto_vote' and ep==4:raise KeyboardInterrupt()
                return original(obj,ep)
            with patch('lfba_comparison.runner.assemble',side_effect=paired_assemble), patch.object(ComparisonTrainer,'_configure_attack',interrupt):
                with self.assertRaises(KeyboardInterrupt):run_stability(self.options(part))
            self.assertTrue(list(part.rglob('*.pth.tar')))
            options=self.options(part);options.resume=True
            with patch('lfba_comparison.runner.assemble',side_effect=paired_assemble):
                self.assertEqual(run_stability(options),0)
            for case in read_json(full/'protocol.json')['cases']:
                for method in ('auto_vote','lfba_original'):
                    suffix=Path(case['id'])/'seed_100'/method
                    a,b=read_json(full/suffix/'metrics.json'),read_json(part/suffix/'metrics.json')
                    for key in ('test','anchor','target_class','final_model_digest','first_prefix','start_state'):
                        self.assertEqual(a[key],b[key])
                    self.assertEqual(read_json(full/suffix/'experiment_data.json')['plans'],read_json(part/suffix/'experiment_data.json')['plans'])

    def test_global_plot_failure_preserves_models_and_report_only_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch'
            with patch('lfba_comparison.runner.assemble',side_effect=paired_assemble), \
                 patch('lfba_comparison.stability_report._parameter_plot',side_effect=RuntimeError('plot failed')):
                with self.assertRaisesRegex(RuntimeError,'plot failed'):
                    run_stability(self.options(root))
            self.assertTrue(list(root.rglob('*.pth.tar')))
            self.assertFalse(read_json(root/'summary.json')['all_complete'])
            with patch('lfba_comparison.runner.assemble',side_effect=AssertionError('no models')):
                self.assertEqual(report_stability(root),0)
            self.assertFalse(list(root.rglob('*.pth.tar')))

    def test_failure_continues_other_configurations_and_resume_repairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch';options=self.options(root)
            def fail(*args,**kwargs):
                if args[0].poison_rate==.1 and args[2]=='auto_vote':
                    raise RuntimeError('one configuration failed')
                return paired_assemble(*args,**kwargs)
            with patch('lfba_comparison.runner.assemble',side_effect=fail):
                self.assertEqual(run_stability(options),1)
            summary=read_json(root/'summary.json')
            self.assertEqual((summary['completed_pairs'],summary['failed_pairs'],summary['missing_pairs']),(1,1,0))
            self.assertEqual(summary['cases'][0]['paired']['delta_acc']['n'],0)
            self.assertTrue(list(root.rglob('*.pth.tar')))
            options.resume=True
            with patch('lfba_comparison.runner.assemble',side_effect=paired_assemble):
                self.assertEqual(run_stability(options),0)
            self.assertFalse(list(root.rglob('*.pth.tar')))

    def test_early_interruption_report_only_does_not_occupy_future_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch';options=self.options(root)
            with patch('lfba_comparison.stability.run_experiment',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):run_stability(options)
            self.assertEqual(report_stability(root),1)
            protocol=read_json(root/'protocol.json')
            self.assertTrue(all(not (root/case['id']).exists() for case in protocol['cases']))
            options.resume=True
            with patch('lfba_comparison.runner.assemble',side_effect=paired_assemble):
                self.assertEqual(run_stability(options),0)

    def test_child_protocol_conflict_cannot_be_overridden_by_complete_metrics(self):
        from lfba_comparison.common import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch';options=self.options(root)
            with patch('lfba_comparison.runner.assemble',side_effect=paired_assemble):
                self.assertEqual(run_stability(options),0)
            case=read_json(root/'protocol.json')['cases'][0]
            folder=root/case['id'];protocol=read_json(folder/'protocol.json')
            protocol['batch_size']=64;write_json(folder/'protocol.json',protocol)
            recovery=folder/'seed_100/auto_vote/.runtime/last_checkpoint.pth.tar'
            recovery.parent.mkdir();recovery.write_bytes(b'recovery')
            options.resume=True
            with patch('lfba_comparison.runner.assemble',side_effect=AssertionError('no models')):
                self.assertEqual(run_stability(options),1)
            self.assertFalse(read_json(root/'summary.json')['all_complete'])
            self.assertTrue(recovery.exists())

    def test_report_only_rejects_changed_matrix_paths_without_writing_outside(self):
        from lfba_comparison.common import write_json
        from lfba_comparison.stability_config import stability_protocol
        from lfba_comparison.stability_report import write_stability_report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch';options=self.options(root)
            protocol=stability_protocol(options)
            protocol['cases'][0]['id']='../outside'
            write_json(root/'protocol.json',protocol)
            with self.assertRaises(ValueError):report_stability(root)
            with self.assertRaisesRegex(ValueError,'directly inside'):
                write_stability_report(root)
            self.assertFalse((Path(tmp)/'outside').exists())


if __name__=='__main__':unittest.main()
