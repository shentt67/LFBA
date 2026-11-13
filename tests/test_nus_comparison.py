"""NUS-WIDE comparison preflight and attacker-side configuration, no training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.compare_lfba_auto_vote import dry_run, main, parse_options
from lfba_comparison.runner import assemble, close, configuration, frozen_protocol
from main import trigger_dimensions

LABELS = ['buildings', 'grass', 'animal', 'water', 'person']


class NusComparisonTests(unittest.TestCase):
    def prepare(self, root):
        data = root / 'data' / 'NUS_WIDE'
        for name in ('Groundtruth/TrainTestLabels', 'Low_Level_Features', 'NUS_WID_Tags'):
            (data / name).mkdir(parents=True)
        for split, count in [('Train', 1001), ('Test', 201)]:
            labels = np.eye(5, dtype=int)[np.arange(count) % 5]
            # Discard multi-label and unlabelled examples, as the real loader does.
            labels[-2] = 1
            labels[-1] = 0
            for i, name in enumerate(LABELS):
                np.savetxt(data / 'Groundtruth/TrainTestLabels' / f'Labels_{name}_{split}.txt', labels[:, i], fmt='%d')
            # Preflight must only inspect the manifest, not parse feature tables.
            (data / 'Low_Level_Features' / f'{split}_Normalized_a.dat').write_text('unread feature placeholder')
            (data / 'NUS_WID_Tags' / f'{split}_Tags1k.dat').write_text('unread tag placeholder')
        return root / 'data'

    def options(self, root, *extra):
        return parse_options(['--dataset', 'NUSWIDE', '--data-dir', str(root / 'data'),
                              '--output', str(root / 'output'), *extra])

    def test_alias_and_separate_default_output(self):
        options = parse_options(['--dataset', 'NUSWID'])
        self.assertEqual(options.dataset, 'NUSWIDE')
        self.assertIn('nuswide', options.output.name)
        self.assertEqual(options.seeds, [2004, 592, 3383, 100])

    def test_preflight_counts_filtered_labels_and_floors_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepare(root)
            with patch.dict('lfba_comparison.policy.ANCHORS', NUSWIDE=0), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(dry_run(self.options(root)), 0)
            report = json.loads(out.getvalue())
            self.assertEqual([report[k] for k in ('train_size', 'validation_size', 'test_size')], [899, 100, 199])
            self.assertEqual(report['candidates'], 89)
            self.assertEqual(report['replacements_per_epoch'], 44)
            self.assertEqual(report['formal_budgets'], {'lfba_original': 1716, 'auto_vote': 1628})
            self.assertEqual(report['tasks'], 8)

    def test_smoke_budget_and_fixed_anchor_retention_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepare(root)
            options = self.options(root, '--smoke', '--seeds', '2004')
            with patch.dict('lfba_comparison.policy.ANCHORS', NUSWIDE=700), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(dry_run(options), 0)
                args = configuration(options, 2004, 'auto_vote', root / 'output')
            report = json.loads(out.getvalue())
            self.assertEqual([report[k] for k in ('train_size', 'validation_size', 'test_size')], [450, 50, 100])
            self.assertEqual(report['budgets'], {'lfba_original': 88, 'auto_vote': 44})
            self.assertTrue(args.retain_anchor)
            self.assertEqual(args.anchor_idx, 700)

    def test_trigger_and_protocol_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepare(root)
            options = self.options(root)
            args = configuration(options, 2004, 'capture', root / 'output')
            self.assertEqual(args.anchor_idx, 33930)
            self.assertEqual(args.poison_dimensions, 10)
            dims = trigger_dimensions(args, None)
            self.assertEqual(len(set(dims)), 10)
            self.assertTrue(all(634 <= d < 1634 for d in dims))
            protocol = frozen_protocol(options)
            self.assertEqual(protocol['selected_labels'], LABELS)
            self.assertEqual(protocol['poison_dimensions'], 10)
            self.assertEqual(protocol['feature_split'], [634, 1000])

    def test_missing_files_are_nuswide_not_har(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(dry_run(self.options(Path(tmp))), 2)
        self.assertIn('NUS_WIDE', out.getvalue())
        self.assertNotIn('UCI HAR', out.getvalue())

    def test_invalid_labels_and_out_of_range_anchor_fail_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = self.prepare(root)
            with self.assertRaisesRegex(ValueError, 'anchor'):
                dry_run(self.options(root))
            file = data / 'NUS_WIDE/Groundtruth/TrainTestLabels/Labels_grass_Train.txt'
            file.write_text('1\n0\n')
            with self.assertRaisesRegex(ValueError, 'rows'):
                dry_run(self.options(root))

    def test_report_rebuild_does_not_require_input_data_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('lfba_comparison.report.write_report') as report, \
                    patch('lfba_comparison.runner.cleanup_completed_batch'), \
                    patch('lfba_comparison.runner.run_experiment') as train:
                self.assertEqual(main(['--dataset', 'NUSWIDE', '--report-only',
                    '--data-dir', str(Path(tmp) / 'missing'), '--output', str(Path(tmp) / 'output')]), 0)
            report.assert_called_once(); train.assert_not_called()

    def test_stability_still_rejects_nuswide(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_options(['--dataset', 'NUSWIDE', '--stability'])

    def test_nus_model_inputs_and_replacement_preserve_honest_features(self):
        def synthetic_features(root, labels, limit, split):
            count = 999 if split == 'Train' else 199
            rng = np.random.default_rng(count)
            return rng.normal(size=(count, 634)), rng.normal(size=(count, 1000)), np.arange(count) % 5

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepare(root)
            options = self.options(root, '--device', '-1')
            with patch.dict('lfba_comparison.policy.ANCHORS', NUSWIDE=0), \
                    patch('dataset.dataset.get_labeled_data', side_effect=synthetic_features):
                trainer, handler = assemble(options, 2004, 'capture', root / 'capture')
            try:
                view = trainer.train_loader.dataset
                recipient, donor = view.ids[:2]
                before = view.data_p.clone()
                labels = view.targets.clone()
                dims = trainer.trigger_dimensions
                view.set_attack({recipient: donor}, trainer.args, dims)
                expected = before[donor, 634:].clone()
                expected[[d-634 for d in dims]] = 1.0
                torch.testing.assert_close(view.data[recipient, :634], before[recipient, :634])
                torch.testing.assert_close(view.data[recipient, 634:], expected)
                torch.testing.assert_close(view.data_p, before)
                torch.testing.assert_close(view.targets, labels)
                x = view.data[[recipient, donor]].float()
                top, honest, attacker = trainer.model_list
                with torch.no_grad():
                    logits = top([honest(x[:, :634]), attacker(x[:, 634:])])
                self.assertEqual(tuple(logits.shape), (2, 5))
                self.assertTrue(torch.isfinite(logits).all())
            finally:
                close(trainer, handler)


if __name__ == '__main__':
    unittest.main()
