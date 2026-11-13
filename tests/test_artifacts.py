import tempfile
import unittest
from pathlib import Path
from tests.helpers import make_trainer

class ArtifactTests(unittest.TestCase):
    def test_training_only_uses_temporary_checkpoint_and_records_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp,attack='LFBA',epoch=2)
            trainer.train()
            self.assertTrue((Path(tmp)/'.runtime'/'last_checkpoint.pth.tar').exists())
            self.assertTrue((Path(tmp)/'experiment_data.json').exists())
            self.assertFalse((Path(tmp)/'last_checkpoint.pth.tar').exists())

    def test_main_complete_keeps_four_files_and_no_runtime(self):
        from unittest.mock import patch
        from tests.test_entry import small_data
        from main import main
        from utils.config import parse_args
        with tempfile.TemporaryDirectory() as tmp, patch('main.build_datasets',side_effect=small_data):
            main(parse_args(['--dataset','PHISHING','--target_label','1','--epoch','1','--device','-1','--results_dir',tmp]))
            self.assertEqual({p.name for p in Path(tmp).iterdir()},
                             {'experiment.log','config.json','experiment_data.json','metrics.json'})

    def test_failed_final_write_retains_runtime(self):
        from unittest.mock import patch
        from tests.test_entry import small_data
        from main import main
        from utils.config import parse_args
        with tempfile.TemporaryDirectory() as tmp, patch('main.build_datasets',side_effect=small_data):
            with patch('main.verify_saved_result',side_effect=OSError('write failed')):
                with self.assertRaises(OSError):
                    main(parse_args(['--dataset','PHISHING','--target_label','1','--epoch','1','--device','-1','--results_dir',tmp]))
            self.assertTrue((Path(tmp)/'.runtime'/'last_checkpoint.pth.tar').exists())
