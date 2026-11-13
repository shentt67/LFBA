import tempfile
import unittest

from tests.helpers import make_trainer


class TrainingTests(unittest.TestCase):
    def test_every_epoch_updates_batchnorm_and_eval_has_no_grad(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(tmp)
            trainer.train()
            bottom = trainer.model_list[1]
            self.assertEqual(int(bottom.bn.num_batches_tracked), 10)
            self.assertTrue(all(not grad for training, grad in bottom.calls if not training))


if __name__ == '__main__':
    unittest.main()
