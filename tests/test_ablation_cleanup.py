"""Real CPU training checks for durable ablation results and temporary models."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from dataset.experiment import DataView
from lfba_comparison.common import read_json, write_json
from lfba_comparison.core_trainer import CoreTrainer
from tests.helpers import make_trainer, TinyData
from utils.utils import set_seed


def tiny_assemble(options, seed, method, output, checkpoint=None, choice=None):
    set_seed(seed)
    base = make_trainer(output, attack='LFBA', epoch=3 if method == 'capture' else (5 if options.smoke else 40),
                        seed=seed, poison_rate=.1, early_stop=0, pretrain_stage=3,
                        comparison_method=method)
    ids = list(range(449)) + [23470]
    loader = DataLoader(DataView(TinyData(23471), ids), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed+1))
    trainer = CoreTrainer(base.device, base.model_list, [], None, base.optimizer_list,
                          base.criterion, loader, base.test_loader, base.test_asr_loader,
                          base.trigger_dimensions, base.logger, base.args, checkpoint,
                          base.val_loader, base.val_asr_loader, {'train_ids': ids}, choice=choice)
    return trainer, None


@contextmanager
def tiny_models():
    torch.set_num_threads(1)
    with ExitStack() as stack:
        for module in ('core_runner', 'anchor_vote_runner'):
            stack.enter_context(patch('lfba_comparison.'+module+'.assemble', side_effect=tiny_assemble))
            stack.enter_context(patch('lfba_comparison.'+module+'.close'))
        yield


class AblationCleanupTests(unittest.TestCase):
    def options(self, kind, root, extra=()):
        if kind == 'core':
            from scripts.run_core_ablation import parse_options
        else:
            from scripts.run_anchor_vote_ablation import parse_options
        return parse_options(['--run', '--smoke', '--device', '-1', '--seeds', '2004',
                              '--output', str(root), *extra])

    def runner(self, kind):
        if kind == 'core':
            from lfba_comparison.core_runner import run
        else:
            from lfba_comparison.anchor_vote_runner import run
        return run

    def reporter(self, kind):
        if kind == 'core':
            from lfba_comparison.core_report import write_report
        else:
            from lfba_comparison.anchor_vote_report import write_report
        return write_report

    def test_success_removes_models_and_completed_resume_never_trains(self):
        for kind in ('core', 'anchor'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, tiny_models():
                root = Path(directory)/'batch'
                options = self.options(kind, root)
                self.assertEqual(self.runner(kind)(options), 0)
                before = read_json(root/'summary.json')['records']
                self.assertFalse(list(root.rglob('.runtime')))
                self.assertFalse(list(root.rglob('last_checkpoint.pth.tar')))
                self.assertTrue((root/'seed_2004/completion_proof.json').is_file())
                self.assertTrue(self.reporter(kind)(root)['all_complete'])
                self.assertEqual(before, read_json(root/'summary.json')['records'])
                options.resume = True
                with patch.object(CoreTrainer, 'train', side_effect=AssertionError('unexpected training')):
                    self.assertEqual(self.runner(kind)(options), 0)

    def test_keep_checkpoints_preserves_models(self):
        for kind in ('core', 'anchor'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, tiny_models():
                root = Path(directory)/'batch'
                options = self.options(kind, root, ['--keep-checkpoints'])
                self.assertEqual(self.runner(kind)(options), 0)
                self.assertTrue(list(root.rglob('.runtime/last_checkpoint.pth.tar')))
                self.assertTrue(self.reporter(kind)(root)['all_complete'])

    def test_plot_failure_retains_models_and_report_only_finishes_cleanup(self):
        for kind, module in (('core', 'core_report'), ('anchor', 'anchor_vote_report')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, tiny_models():
                root = Path(directory)/'batch'
                options = self.options(kind, root)
                import importlib
                report_module = importlib.import_module('lfba_comparison.'+module)
                original_plot = report_module.plot
                required = 4 if kind == 'core' else 2
                def fail_final_plot(path, rows, protocol):
                    if len({r['method'] for r in rows}) == required:
                        raise OSError('plot failed')
                    return original_plot(path, rows, protocol)
                with patch('lfba_comparison.'+module+'.plot', side_effect=fail_final_plot):
                    with self.assertRaisesRegex(OSError, 'plot failed'):
                        self.runner(kind)(options)
                self.assertTrue(list(root.rglob('.runtime/last_checkpoint.pth.tar')))
                options.resume = True
                with patch.object(CoreTrainer, 'train', side_effect=AssertionError('unexpected training')):
                    self.assertEqual(self.runner(kind)(options), 0)
                self.assertFalse(list(root.rglob('.runtime')))

    def test_proof_rejects_modified_records_and_missing_proof(self):
        for kind, method in (('core', 'auto_vote'), ('anchor', 'fixed_vote')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, tiny_models():
                root = Path(directory)/'batch'
                self.assertEqual(self.runner(kind)(self.options(kind, root)), 0)
                data_path = root/'seed_2004'/method/'experiment_data.json'
                original = read_json(data_path)
                altered = read_json(data_path)
                altered['plans'][0]['donors'][0] = -1
                write_json(data_path, altered)
                self.assertFalse(self.reporter(kind)(root)['all_complete'])
                write_json(data_path, original)
                self.assertTrue(self.reporter(kind)(root)['all_complete'])
                (root/'seed_2004/completion_proof.json').unlink()
                self.assertFalse(self.reporter(kind)(root)['all_complete'])

    def test_anchor_completed_cli_resume_bypasses_data_and_source_preflight(self):
        from scripts.run_anchor_vote_ablation import main
        with tempfile.TemporaryDirectory() as directory, tiny_models():
            root = Path(directory)/'batch'
            self.assertEqual(self.runner('anchor')(self.options('anchor', root)), 0)
            with patch('lfba_comparison.anchor_vote_runner.preflight', side_effect=AssertionError('no preflight')):
                self.assertEqual(main(['--run', '--resume', '--smoke', '--device', '-1',
                                       '--seeds', '2004', '--output', str(root)]), 0)

    def test_report_only_cleans_after_plot_recovery_and_is_idempotent(self):
        from lfba_comparison.completion import finish_ablation
        for kind in ('core', 'anchor'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, tiny_models():
                root = Path(directory)/'batch'
                # Suppress finalization to leave a fully audited batch with temporary models.
                runner_module = 'core_runner' if kind == 'core' else 'anchor_vote_runner'
                with patch('lfba_comparison.'+runner_module+'.finish_ablation'):
                    self.assertEqual(self.runner(kind)(self.options(kind, root)), 0)
                if kind == 'core':
                    from scripts.run_core_ablation import main
                else:
                    from scripts.run_anchor_vote_ablation import main
                with patch.object(CoreTrainer, 'train', side_effect=AssertionError('unexpected training')):
                    self.assertEqual(main(['--report-only', '--output', str(root)]), 0)
                    self.assertEqual(main(['--report-only', '--output', str(root)]), 0)
                self.assertFalse(list(root.rglob('.runtime')))
                finish_ablation(root, read_json(root/'summary.json'))

    def test_failed_proof_write_preserves_checkpoints(self):
        from lfba_comparison.completion import write_json as real_write
        for kind in ('core', 'anchor'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, tiny_models():
                root = Path(directory)/'batch'
                def fail_proof(path, value):
                    if Path(path).name == 'completion_proof.json':
                        raise OSError('proof disk failure')
                    return real_write(path, value)
                with patch('lfba_comparison.completion.write_json', side_effect=fail_proof):
                    with self.assertRaisesRegex(OSError, 'proof disk failure'):
                        self.runner(kind)(self.options(kind, root))
                self.assertTrue(list(root.rglob('.runtime/last_checkpoint.pth.tar')))

    def test_evaluation_or_metrics_failure_preserves_recovery_files(self):
        original_test = CoreTrainer.test
        from lfba_comparison.core_runner import write_json as real_write
        for kind, method in (('core', 'auto_single'), ('anchor', 'fixed_single')):
            for failure in ('evaluation', 'metrics'):
                with self.subTest(kind=kind, failure=failure), tempfile.TemporaryDirectory() as directory, tiny_models():
                    root = Path(directory)/'batch'
                    calls = 0
                    def fail_independent_evaluation(obj, ep=0, split='test'):
                        nonlocal calls
                        if obj.method == method and split == 'test':
                            calls += 1
                            if calls == 2:
                                raise RuntimeError('independent evaluation failed')
                        return original_test(obj, ep, split)
                    def fail_metrics_write(path, value):
                        if Path(path).parent.name == method and Path(path).name == 'metrics.json':
                            raise OSError('metrics disk failure')
                        return real_write(path, value)
                    target = (patch.object(CoreTrainer, 'test', fail_independent_evaluation) if failure == 'evaluation'
                              else patch('lfba_comparison.core_runner.write_json', side_effect=fail_metrics_write))
                    with target:
                        self.assertEqual(self.runner(kind)(self.options(kind, root)), 1)
                    self.assertTrue((root/'seed_2004'/method/'.runtime/last_checkpoint.pth.tar').exists())
                    self.assertFalse((root/'seed_2004/completion_proof.json').exists())

    def test_failed_proof_readback_preserves_all_models(self):
        from lfba_comparison.completion import write_json as real_write
        with tempfile.TemporaryDirectory() as directory, tiny_models():
            root = Path(directory)/'batch'
            def corrupt_proof_write(path, value):
                if Path(path).name == 'completion_proof.json':
                    value = dict(value, seed=99)
                return real_write(path, value)
            with patch('lfba_comparison.completion.write_json', side_effect=corrupt_proof_write):
                with self.assertRaisesRegex(ValueError, 'completion proof'):
                    self.runner('core')(self.options('core', root))
            self.assertEqual(len(list(root.rglob('.runtime/last_checkpoint.pth.tar'))), 5)

    def test_cleanup_validates_all_branches_before_deleting_any_model(self):
        from lfba_comparison.completion import finish_ablation
        for kind, method in (('core', 'auto_vote'), ('anchor', 'fixed_vote')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, tiny_models():
                root = Path(directory)/'batch'
                runner_module = 'core_runner' if kind == 'core' else 'anchor_vote_runner'
                with patch('lfba_comparison.'+runner_module+'.finish_ablation'):
                    self.assertEqual(self.runner(kind)(self.options(kind, root)), 0)
                models = list(root.rglob('.runtime/last_checkpoint.pth.tar'))
                config_path = root/'seed_2004'/method/'config.json'
                config = read_json(config_path)
                config['target_label'] = 99
                write_json(config_path, config)
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    finish_ablation(root, read_json(root/'summary.json'))
                self.assertTrue(all(p.exists() for p in models))

    def test_interrupted_branch_resumes_to_the_same_model_then_cleans(self):
        from utils.checkpoint import atomic_save
        from utils.artifacts import runtime_checkpoint
        for kind, method in (('core', 'auto_single'), ('anchor', 'fixed_single')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, tiny_models():
                full, partial = Path(directory)/'full', Path(directory)/'partial'
                self.assertEqual(self.runner(kind)(self.options(kind, full)), 0)
                original = CoreTrainer._snapshot
                def interrupt(obj):
                    snapshot = original(obj)
                    if obj.method == method and obj.completed_epoch == 4:
                        atomic_save(snapshot, runtime_checkpoint(obj.args.results_dir))
                        raise InterruptedError('interrupted')
                    return snapshot
                with patch.object(CoreTrainer, '_snapshot', interrupt):
                    self.assertEqual(self.runner(kind)(self.options(kind, partial)), 1)
                self.assertTrue(runtime_checkpoint(partial/'seed_2004'/method).exists())
                options = self.options(kind, partial); options.resume = True
                self.assertEqual(self.runner(kind)(options), 0)
                for full_metrics in (full/'seed_2004').glob('*/metrics.json'):
                    left = read_json(full_metrics)
                    right = read_json(partial/'seed_2004'/full_metrics.parent.name/'metrics.json')
                    self.assertEqual(left['test'], right['test'])
                    self.assertEqual(left['final_model_digest'], right['final_model_digest'])
                self.assertFalse(list(partial.rglob('.runtime')))

    def test_formal_anchor_report_and_cli_resume_survive_source_removal(self):
        from scripts.run_core_ablation import parse_options as core_options
        from scripts.run_anchor_vote_ablation import parse_options as anchor_options, main
        from lfba_comparison.anchor_vote_runner import file_hash
        from utils.artifacts import runtime_checkpoint
        from utils.checkpoint import load_checkpoint
        from lfba_comparison.common import digest
        with tempfile.TemporaryDirectory() as directory, tiny_models():
            root = Path(directory)/'target'
            with tempfile.TemporaryDirectory() as source_directory:
                source = Path(source_directory)/'source'
                core = core_options(['--run', '--device', '-1', '--seeds', '2004',
                                     '--keep-checkpoints', '--output', str(source)])
                self.assertEqual(self.runner('core')(core), 0)
                folder = source/'seed_2004'
                selection = read_json(folder/'selection.json')
                prefix_path = runtime_checkpoint(folder/'clean_prefix')
                prefix = load_checkpoint(prefix_path)
                references = {m:dict(directory=str(folder/m), choice=selection['choices'][m])
                              for m in ('auto_rank_last', 'auto_vote', 'auto_mean')}
                hashes = {str(p):file_hash(p) for p in source.rglob('*') if p.is_file()}
                manifest = dict(protocol_path=str(source/'protocol.json'), files=hashes,
                                seeds={'2004':dict(prefix_path=str(prefix_path), prefix_digest=digest(prefix),
                                                  prefix_file_hash=file_hash(prefix_path), references=references,
                                                  data_digest=digest(prefix['data_state']))})
                options = anchor_options(['--run', '--device', '-1', '--seeds', '2004',
                                          '--source-results', str(source), '--output', str(root)])
                self.assertEqual(self.runner('anchor')(options, manifest), 0)
                self.assertEqual(hashes, {str(p):file_hash(p) for p in source.rglob('*') if p.is_file()})
                before = read_json(root/'summary.json')
                from utils.artifacts import cleanup_runtime
                from lfba_comparison.anchor_vote_runner import inspect_sources
                cleanup_runtime(folder/'clean_prefix')
                with self.assertRaisesRegex(ValueError, 'keep-checkpoints'):
                    inspect_sources(options)
            self.assertFalse(source.exists())
            self.assertTrue(self.reporter('anchor')(root)['all_complete'])
            self.assertEqual(before, read_json(root/'summary.json'))
            with patch.object(CoreTrainer, 'train', side_effect=AssertionError('unexpected training')):
                self.assertEqual(main(['--run', '--resume', '--device', '-1', '--seeds', '2004',
                                       '--source-results', str(source), '--output', str(root)]), 0)


class RuntimeCleanupTests(unittest.TestCase):
    def test_registered_legacy_files_removed_unknown_files_preserved(self):
        from utils.artifacts import cleanup_runtime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in (root, root/'.runtime', root/'runtime'):
                folder.mkdir(exist_ok=True)
                for name in ('last_checkpoint.pth.tar', 'best_checkpoint.pth.tar', 'check_point.tar',
                             'last_checkpoint.pth.tar.tmp', 'check_point.tar.tmp'):
                    (folder/name).write_bytes(b'model')
            (root/'runtime/unknown.bin').write_bytes(b'preserve')
            cleanup_runtime(root)
            cleanup_runtime(root)
            self.assertEqual({str(p.relative_to(root)) for p in root.rglob('*')},
                             {'runtime', str(Path('runtime')/'unknown.bin')})
