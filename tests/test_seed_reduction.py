import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.compare_lfba_auto_vote import parse_options
from lfba_comparison.common import read_json,write_json
from lfba_comparison.stability_config import stability_protocol,case_options
from lfba_comparison.runner import frozen_protocol

class SeedReductionTests(unittest.TestCase):
    def setup_batch(self,root):
        opts=parse_options(['--stability','--run','--resume','--output',str(root)])
        old=stability_protocol(opts);write_json(root/'protocol.json',old)
        child=case_options(opts,old['cases'][0]);write_json(child.output/'protocol.json',frozen_protocol(child))
        kept=child.output/'seed_2004/auto_vote/metrics.json';write_json(kept,{'sentinel':1})
        extra=child.output/'seed_101/.runtime/state';extra.parent.mkdir(parents=True);extra.write_bytes(b'keep')
        opts.seeds=opts.seeds[:4]
        return opts,child.output,kept,extra

    def test_reduce_preserves_results_and_extra_state(self):
        from lfba_comparison.seed_reduction import reduce_seeds
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);opts,child,kept,extra=self.setup_batch(root)
            reduce_seeds(root,opts,stability_protocol(opts))
            self.assertEqual(read_json(root/'protocol.json')['seeds'],opts.seeds)
            self.assertEqual(read_json(child/'protocol.json')['seeds'],opts.seeds)
            self.assertEqual(read_json(kept),{'sentinel':1});self.assertEqual(extra.read_bytes(),b'keep')
            self.assertTrue(read_json(root/'seed_reduction.json')['complete'])
            reduce_seeds(root,opts,stability_protocol(opts))

    def test_conflict_changes_nothing(self):
        from lfba_comparison.seed_reduction import reduce_seeds
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);opts,child,_,_=self.setup_batch(root)
            saved=read_json(child/'protocol.json');saved['lr']=.1;write_json(child/'protocol.json',saved)
            with self.assertRaises(ValueError):reduce_seeds(root,opts,stability_protocol(opts))
            self.assertEqual(len(read_json(root/'protocol.json')['seeds']),5)

    def test_interrupted_migration_can_retry(self):
        from lfba_comparison.seed_reduction import reduce_seeds
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);opts,child,_,_=self.setup_batch(root)
            def fail(path,value):
                if path==root/'protocol.json':raise OSError('interrupted')
                write_json(path,value)
            with patch('lfba_comparison.seed_reduction.write_json',side_effect=fail):
                with self.assertRaises(OSError):reduce_seeds(root,opts,stability_protocol(opts))
            reduce_seeds(root,opts,stability_protocol(opts))
            self.assertEqual(read_json(root/'protocol.json')['seeds'],opts.seeds)

    def test_flag_requires_resume_and_explicit_seeds(self):
        with self.assertRaises(SystemExit):parse_options(['--stability','--reduce-seeds'])
        opts=parse_options(['--stability','--run','--resume','--reduce-seeds','--seeds','2004','592','3383','100'])
        self.assertTrue(opts.reduce_seeds)

    def test_running_batch_lock_blocks_reduction(self):
        from lfba_comparison.common import batch_lock
        from lfba_comparison.stability import run_stability
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);opts,_,_,_=self.setup_batch(root);opts.reduce_seeds=True
            with batch_lock(root):
                with self.assertRaisesRegex(RuntimeError,'batch lock'):run_stability(opts)
            self.assertEqual(len(read_json(root/'protocol.json')['seeds']),5)
