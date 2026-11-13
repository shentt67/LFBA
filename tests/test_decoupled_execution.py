"""Manual-run guardrails, durable provenance and synthetic state restoration."""
import copy
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lfba_comparison.common import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


class InterfaceTests(unittest.TestCase):
    def test_default_is_read_only_and_formal_matrix_is_frozen(self):
        from scripts.run_decoupled_selection import parse_options, preflight
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            args = parse_options(['--output', str(output)])
            report = preflight(args)
            self.assertFalse(args.run)
            self.assertFalse(report['training_started'])
            self.assertEqual(report['tasks'], 16)
            self.assertEqual(report['reference_count'], 4500)
            self.assertEqual([r['cumulative_replacements'] for r in report['cases']], [8325, 41625])
            self.assertFalse(output.exists())
            self.assertEqual(report['historical_reference_count'], 16)
            self.assertEqual(report['errors'], [])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): parse_options(['--resume'])
            with self.assertRaises(SystemExit): parse_options(['--output', str(ROOT / 'data')])
            with self.assertRaises(SystemExit): parse_options(['--output', str(ROOT / 'results/lfba_stability_full/new')])

    def test_report_only_cannot_enter_runner_or_modify_inputs(self):
        from scripts.run_decoupled_selection import main, parse_options
        from lfba_comparison.decoupled_protocol import protocol
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_options(['--smoke', '--output', str(root)])
            write_json(root / 'protocol.json', protocol(args))
            with patch('lfba_comparison.decoupled_runner.run', side_effect=AssertionError('training forbidden')):
                self.assertEqual(main(['--report-only', '--output', str(root)]), 1)
            summary = read_json(root / 'summary.json')
            self.assertEqual(summary['complete'], 0)
            self.assertFalse(summary['all_complete'])
            self.assertEqual(summary['expected'], 4)
            self.assertFalse((root / 'seed_2004').exists())

    def test_invalid_protocol_revokes_previous_success_receipt(self):
        from lfba_comparison.decoupled_report import write_report
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / 'verification.json', dict(status='complete', original_inputs_unchanged=True))
            write_json(root / 'summary.json', dict(all_complete=True, method_passed={'decoupled_mean': True},
                                                   acceptance=[dict(p=.01, method='decoupled_mean', passed=True)]))
            write_json(root / 'acceptance.json', [dict(passed=True)])
            (root / 'report.md').write_text('previous success', encoding='utf-8')
            (root / 'protocol.json').write_text('{broken', encoding='utf-8')
            with self.assertRaises(ValueError): write_report(root)
            self.assertEqual(read_json(root / 'verification.json')['status'], 'failed')
            self.assertFalse(read_json(root / 'acceptance.json')[0]['passed'])
            self.assertNotIn('previous success', (root / 'report.md').read_text(encoding='utf-8'))


class ReferenceTests(unittest.TestCase):
    def test_repeated_wrong_exposures_are_events_not_unique_recipients(self):
        from lfba_comparison.decoupled_records import exposure_diagnostics
        stats = exposure_diagnostics([0, 0, 1, 0, 1, 1], dict(anchor_id=0, candidates=[1, 2, 3]),
            [dict(recipients=[1, 2], donors=[4, 5]), dict(recipients=[2, 3], donors=[4, 5])])
        self.assertEqual(stats['wrong_exposures'], 2)
        self.assertEqual(stats['total_exposures'], 4)
        self.assertEqual(stats['unique_recipients'], 3)
        self.assertEqual(stats['max_reuse'], 2)
        self.assertEqual(stats['recipient_coverage'], 1.)
        self.assertEqual(stats['recipient_purity'], .5)
        self.assertEqual(stats['donor_class_counts'], {'1': 4})

    def test_historical_references_require_identity_and_clean_history(self):
        from lfba_comparison.decoupled_records import load_references, verify_reference_prefix
        refs = load_references(ROOT / 'results/lfba_stability_full', 2004)
        self.assertEqual(len(refs), 4)
        auto = next(r for r in refs if r['method'] == 'auto_vote')
        record = auto['record']
        prefix = dict(data_state=auto['data_state'], trigger_dimensions=[], history=auto['clean_history'],
                      comparison=dict(initial_digest=record['initial_digest'], prefix=[record['first_prefix']]))
        proof = verify_reference_prefix(prefix, refs)
        self.assertTrue(proof['verified'])
        self.assertFalse(proof['rounds_2_3_full_state_available'])
        changed = copy.deepcopy(prefix)
        changed['comparison']['initial_digest'] = 'wrong'
        with self.assertRaises(ValueError): verify_reference_prefix(changed, refs)
        changed = copy.deepcopy(prefix)
        changed['history'][1]['train_loss'] += .01
        with self.assertRaises(ValueError): verify_reference_prefix(changed, refs)

    def test_input_receipt_detects_change(self):
        from lfba_comparison.decoupled_records import snapshot, verify_manifest
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'original.json'
            p.write_text('{}', encoding='utf-8')
            manifest = snapshot([p])
            self.assertTrue(verify_manifest(manifest)['unchanged'])
            p.write_text('{"changed": true}', encoding='utf-8')
            self.assertFalse(verify_manifest(manifest)['unchanged'])


class StateTests(unittest.TestCase):
    def test_synthetic_batch_reports_resumes_and_rejects_tampered_receipts(self):
        import torch
        from torch.utils.data import DataLoader
        from tests.helpers import make_trainer, TinyData
        from dataset.experiment import DataView
        from lfba_comparison.core_trainer import CoreTrainer
        from lfba_comparison.decoupled_runner import run
        from lfba_comparison.decoupled_report import write_report
        from scripts.run_decoupled_selection import parse_options
        from utils.utils import set_seed

        def synthetic(options, seed, method, output, checkpoint=None, choice=None, rate=.1):
            set_seed(seed)
            base = make_trainer(output, attack='LFBA', epoch=3 if method == 'capture' else 5,
                early_stop=0, pretrain_stage=3, poison_rate=rate, comparison_method=method, batch_size=128)
            train = DataLoader(DataView(TinyData(450), range(450)), batch_size=128,
                               shuffle=True, generator=torch.Generator().manual_seed(seed + 1))
            val = DataLoader(TinyData(50), batch_size=50, generator=torch.Generator().manual_seed(seed + 2))
            val_asr = DataLoader(TinyData(50), batch_size=50, generator=torch.Generator().manual_seed(seed + 3))
            test = DataLoader(TinyData(100), batch_size=100, generator=torch.Generator().manual_seed(seed + 4))
            test_asr = DataLoader(TinyData(100), batch_size=100, generator=torch.Generator().manual_seed(seed + 5))
            state = dict(train_ids=list(range(450)), val_ids=list(range(450, 500)), test_ids=list(range(500, 600)))
            obj = CoreTrainer(base.device, base.model_list, [], None, base.optimizer_list, base.criterion,
                train, test, test_asr, [15], base.logger, base.args, checkpoint, val, val_asr, state, choice=choice)
            import logging
            handler = logging.NullHandler()
            obj.logger.addHandler(handler)
            return obj, handler

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'run'
            source = Path(folder) / 'input.json'
            source.write_text('{}', encoding='utf-8')
            source2 = Path(folder) / 'input2.json'
            source2.write_text('{}', encoding='utf-8')
            options = parse_options(['--smoke', '--device', '-1', '--output', str(root)])
            with patch('lfba_comparison.decoupled_runner.assemble', side_effect=synthetic), \
                 patch('lfba_comparison.decoupled_runner.input_files', return_value=[source, source2]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(options), 0)
            report = read_json(root / 'summary.json')
            self.assertEqual(report['complete'], 4)
            self.assertTrue(report['all_complete'])
            self.assertFalse(any(report['method_passed'].values()))
            self.assertEqual(len(list(root.rglob('last_checkpoint.pth.tar'))), 5)
            options.resume = True
            with patch('lfba_comparison.decoupled_runner.assemble', side_effect=AssertionError('Resume must skip completed training')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(options), 0)
            original_manifest = read_json(root / 'input_manifest.json')
            reduced = dict(original_manifest)
            reduced.pop(str(source2.resolve()))
            write_json(root / 'input_manifest.json', reduced)
            incomplete = write_report(root)
            self.assertFalse(incomplete['all_complete'])
            self.assertFalse(incomplete['input_verification']['unchanged'])
            write_json(root / 'input_manifest.json', original_manifest)
            path = root / 'seed_2004/p_0p01_s_0p5/decoupled_mean/metrics.json'
            changed = read_json(path)
            changed['test']['asr'] = (changed['test']['asr'] + .5) % 1.
            write_json(path, changed)
            report = write_report(root)
            self.assertEqual(report['complete'], 3)
            self.assertFalse(report['all_complete'])
            self.assertTrue(report['issues'])
            self.assertEqual(read_json(root / 'verification.json')['status'], 'incomplete')

    def test_new_choices_share_prefix_and_resume_exactly(self):
        import torch
        from tests.helpers import make_trainer
        from lfba_comparison.core_trainer import CoreTrainer
        from lfba_comparison.decoupled_policy import build_ranking, choose
        from lfba_comparison.decoupled_records import verify_final
        from utils.artifacts import runtime_checkpoint
        from utils.checkpoint import load_checkpoint, atomic_save
        from utils.utils import set_seed

        def make(path, method, checkpoint=None, choice=None, p=.1):
            base = make_trainer(path, attack='LFBA', epoch=3 if method == 'capture' else 5,
                                early_stop=0, pretrain_stage=3, poison_rate=p,
                                comparison_method=method)
            state = dict(train_ids=list(range(20)))
            return CoreTrainer(base.device, base.model_list, [], None, base.optimizer_list,
                               base.criterion, base.train_loader, base.test_loader, base.test_asr_loader,
                               base.trigger_dimensions, base.logger, base.args, checkpoint,
                               base.val_loader, base.val_asr_loader, state, choice=choice)

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            set_seed(100)
            make(root / 'prefix', 'capture').train()
            prefix = load_checkpoint(runtime_checkpoint(root / 'prefix'))
            prefix_hash = digest(prefix)
            rankings = build_ranking(prefix['comparison']['observations'], 2)
            starts = []
            for method in ('decoupled_mean', 'decoupled_vote'):
                for p in (.1, .2):
                    selected = choose(rankings, method, int(20 * p))
                    out = root / (method + str(p))
                    obj = make(out, method, prefix, selected, p)
                    obj.train()
                    final = load_checkpoint(runtime_checkpoint(out))
                    verify_final(final, prefix, selected, 5, p)
                    starts.append(obj.start_state)
                    self.assertEqual(obj.cumulative, 2 if p == .1 else 4)
                    self.assertEqual(prefix_hash, digest(prefix))
                    interrupted = make(root / 'partial', method, prefix, selected, p)
                    original = interrupted._snapshot
                    def stop():
                        state = original()
                        if interrupted.completed_epoch == 4:
                            atomic_save(state, runtime_checkpoint(root / 'partial'))
                            raise InterruptedError('synthetic interruption')
                        return state
                    interrupted._snapshot = stop
                    with self.assertRaises(InterruptedError): interrupted.train()
                    resumed = make(root / 'resumed', method,
                                   load_checkpoint(runtime_checkpoint(root / 'partial')), selected, p)
                    resumed.train()
                    self.assertEqual(resumed.plans, obj.plans)
                    self.assertEqual(digest([m.state_dict() for m in resumed.model_list]), digest(final['state_dict']))
                    self.assertEqual(resumed.test(4), obj.test(4))
                    corrupt = copy.deepcopy(final)
                    corrupt['comparison']['plans'][0]['donors'][0] = selected['candidates'][0]
                    with self.assertRaises(ValueError): verify_final(corrupt, prefix, selected, 5, p)
                    corrupt = copy.deepcopy(final)
                    corrupt['comparison']['cumulative'] += 1
                    with self.assertRaises(ValueError): verify_final(corrupt, prefix, selected, 5, p)
            self.assertEqual(len(set(starts)), 1)


if __name__ == '__main__':
    unittest.main()
