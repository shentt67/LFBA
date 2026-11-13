import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import os

import torch
from utils.checkpoint import atomic_save


class CheckpointIOTests(unittest.TestCase):
    def test_persistent_lock_is_bounded_and_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pth'
            atomic_save({'epoch':1},path)
            error=PermissionError(13,'persistent lock')
            error.winerror=32
            with patch('utils.checkpoint.os.replace',side_effect=error) as replace, patch('utils.checkpoint.time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_save({'epoch':2},path)
            self.assertEqual(replace.call_count,8)
            self.assertEqual(torch.load(path,weights_only=True)['epoch'],1)

    def test_transient_windows_file_lock_retries_without_corrupting_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pth'
            atomic_save({'epoch':1},path)
            real_replace=os.replace
            attempts=[]
            def temporarily_locked(source,destination):
                attempts.append(1)
                if len(attempts)==1:
                    error=PermissionError(13,'transient scanner lock')
                    error.winerror=32
                    raise error
                return real_replace(source,destination)
            with patch('utils.checkpoint.os.replace',side_effect=temporarily_locked):
                atomic_save({'epoch':2},path)
            self.assertEqual(torch.load(path,weights_only=True)['epoch'],2)
            self.assertEqual(len(attempts),2)
