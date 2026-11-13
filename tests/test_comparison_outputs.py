"""Exercise batch dispatch and resume without creating models or training."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.compare_lfba_auto_vote import parse_options
from lfba_comparison.common import write_json
from lfba_comparison.policy import METHODS
from lfba_comparison.runner import run_experiment


class ComparisonOutputTests(unittest.TestCase):
    def test_fresh_batch_only_writes_protocol_and_dispatches_two_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'new'
            options = parse_options(['--run', '--seeds', '2004', '--output', str(root)])
            prefix = dict(epoch=1, comparison=dict(method='capture', choice=None, plans=[]),
                          attack_state=dict(current_plan={}))
            capture = SimpleNamespace(train=lambda: None)
            with patch('lfba_comparison.runner.assemble', return_value=(capture, None)), \
                 patch('lfba_comparison.runner.close'), \
                 patch('lfba_comparison.runner.export'), \
                 patch('lfba_comparison.runner.load_checkpoint', return_value=prefix), \
                 patch('lfba_comparison.runner.is_complete', return_value=False), \
                 patch('lfba_comparison.runner.run_task') as task, \
                 patch('lfba_comparison.report.write_report'):
                run_experiment(options)
            self.assertEqual([c.args[2] for c in task.call_args_list], list(METHODS))
            self.assertEqual({p.name for p in root.iterdir()}, {'protocol.json'})

    def test_two_methods_without_source_or_environment_artifacts_and_resume(self):
        self.assertEqual(METHODS, ('lfba_original', 'auto_vote'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'batch'
            options = parse_options(['--run', '--seeds', '2004', '--output', str(root)])
            prefix = dict(epoch=1, comparison=dict(method='capture', choice=None, plans=[]),
                          attack_state=dict(current_plan={}))
            with patch('lfba_comparison.runner.load_checkpoint', return_value=prefix), \
                 patch('lfba_comparison.runner.is_complete', return_value=False), \
                 patch('lfba_comparison.runner.run_task') as task, \
                 patch('lfba_comparison.runner.assemble', side_effect=AssertionError('training forbidden')), \
                 patch('lfba_comparison.report.write_report'):
                clean = root / 'seed_2004' / 'clean_first' / '.runtime'
                clean.mkdir(parents=True)
                (clean / 'last_checkpoint.pth.tar').touch()
                options.resume = True
                from lfba_comparison.runner import frozen_protocol
                write_json(root / 'protocol.json', frozen_protocol(options))
                run_experiment(options)
                self.assertEqual([c.args[2] for c in task.call_args_list], list(METHODS))
                for name in ('source_hashes.json', 'source_snapshot.zip', 'environment.json'):
                    self.assertFalse((root / name).exists())
                # Obsolete source metadata and a changed source root cannot block resume.
                write_json(root / 'source_hashes.json', {'obsolete.py': 'different'})
                options.code_root = Path(directory) / 'changed-source'
                clean.mkdir(parents=True,exist_ok=True)
                (clean/'last_checkpoint.pth.tar').touch()
                run_experiment(options)
                self.assertEqual(task.call_count, 4)


if __name__ == '__main__':
    unittest.main()
