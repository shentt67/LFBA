import tempfile
import unittest
from pathlib import Path


class ConfigTests(unittest.TestCase):
    def test_supported_options_and_explicit_directory(self):
        import utils.config as config
        with tempfile.TemporaryDirectory() as tmp:
            args=config.parse_args(['--results_dir',tmp,'--dataset','NUSWIDET','--pretrain_stage','4'])
        self.assertEqual(args.results_dir,tmp)
        self.assertEqual(args.dataset,'NUSWIDE')
        self.assertEqual(args.similarity,'legacy_l1')
        self.assertEqual(config.attack_start_epoch(args),4)

    def test_invalid_configuration_rejected(self):
        from utils.config import parse_args
        for flags in [ ['--client_num','3'], ['--attack_client_num','0'],
                       ['--select_replace','--poison_all'], ['--dataset','NUSWIDEI'],
                       ['--poison_rate','1.1'], ['--select_rate','nan'] ]:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                parse_args(flags)

    def test_pretrain_delays_actual_poisoning(self):
        from tests.helpers import make_trainer
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp,attack='LFBA',pretrain_stage=4,epoch=3)
            trainer.train()
            self.assertFalse(hasattr(trainer,'poison_indexes') and trainer.poison_indexes is not None)
