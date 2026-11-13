import tempfile
import unittest
from pathlib import Path

import torch

from tests.helpers import make_trainer
from utils.utils import set_seed
from unittest.mock import patch


class CheckpointTests(unittest.TestCase):
    def test_resume_after_poisoning_preserves_unimproved_best(self):
        fixed=dict(acc=.5,asr=.5,poison_acc=.5,target_acc=.5,loss=1.,counts={},undefined={})
        with tempfile.TemporaryDirectory() as tmp:
            set_seed(100)
            full=make_trainer(Path(tmp)/'full',attack='LFBA',epoch=4)
            with patch.object(full,'test',return_value=fixed):
                full.train()
            set_seed(100)
            first=make_trainer(Path(tmp)/'part',attack='LFBA',epoch=2)
            with patch.object(first,'test',return_value=fixed):
                first.train()
            ck=torch.load(Path(tmp)/'part'/'.runtime'/'last_checkpoint.pth.tar',weights_only=True)
            self.assertTrue(ck['attack_state']['current_plan'])
            resumed=make_trainer(Path(tmp)/'resumed',attack='LFBA',epoch=4)
            resumed.checkpoint=ck
            with patch.object(resumed,'test',return_value=fixed):
                resumed.train()
            self.assertEqual(resumed.best_epoch,1)
            self.assertEqual(resumed.no_change,full.no_change)
            self.assertEqual(resumed.current_plan,full.current_plan)
            for a,b in zip(full.model_list,resumed.model_list):
                for k,v in a.state_dict().items():
                    torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
            self.assertFalse(list(Path(tmp).rglob('best_checkpoint.pth.tar')))
            restored_last=torch.load(Path(tmp)/'resumed'/'.runtime'/'last_checkpoint.pth.tar',weights_only=True)
            for a,b in zip(full.best_state_dict,restored_last['best_state_dict']):
                for k,v in a.items():
                    torch.testing.assert_close(v,b[k],rtol=0,atol=0)

    def test_lfba_interrupted_resume_matches_continuous_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            set_seed(100)
            full=make_trainer(Path(tmp)/'full',attack='LFBA',epoch=3)
            full.train()
            set_seed(100)
            first=make_trainer(Path(tmp)/'part',attack='LFBA',epoch=1)
            first.train()
            path=Path(tmp)/'part'/'.runtime'/'last_checkpoint.pth.tar'
            self.assertTrue(path.exists(),'save every completed epoch, not only improvements')
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            resumed=make_trainer(Path(tmp)/'resume',attack='LFBA',epoch=3)
            resumed.checkpoint=checkpoint
            resumed.train()
            self.assertEqual(full.poison_indexes,resumed.poison_indexes)
            self.assertEqual(full.current_plan,resumed.current_plan)
            self.assertEqual(full.best_epoch,resumed.best_epoch)
            self.assertEqual(full.best_score,resumed.best_score)
            self.assertEqual(full.history,resumed.history)
            for a,b in zip(full.model_list,resumed.model_list):
                for k,v in a.state_dict().items():
                    torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)

    def test_empty_asr_denominator_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp)
            trainer.test_loader.dataset.targets[:]=1
            metrics=trainer.test(0)
            self.assertIsNone(metrics['asr'])
            self.assertEqual(metrics['counts']['asr'],0)
            self.assertIn('asr',metrics['undefined'])

    def test_legacy_checkpoint_cannot_claim_strict_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp)
            trainer.checkpoint={'epoch':1,'best_acc':.5}
            with self.assertRaisesRegex(ValueError,'legacy|Legacy'):
                trainer.train()
