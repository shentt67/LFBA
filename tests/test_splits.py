import tempfile
import unittest
import numpy as np

from tests.helpers import make_trainer


class SplitTests(unittest.TestCase):
    def test_fixed_stratified_split_keeps_original_anchor(self):
        from dataset.experiment import split_training_ids
        labels=np.tile(np.arange(2),100)
        train,val=split_training_ids(labels,.1,100,anchor_idx=7)
        self.assertIn(7,train)
        self.assertFalse(set(train)&set(val))
        self.assertEqual((len(train),len(val)),(180,20))
        self.assertEqual((train,val),split_training_ids(labels,.1,100,anchor_idx=7))

    def test_training_never_reads_test_loaders(self):
        class Forbidden:
            def __iter__(self): raise AssertionError('test set used for model selection')
        with tempfile.TemporaryDirectory() as tmp:
            tr=make_trainer(tmp)
            tr.val_loader=tr.test_loader
            tr.val_asr_loader=tr.test_asr_loader
            tr.test_loader=Forbidden()
            tr.test_asr_loader=Forbidden()
            tr.train()
