import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dataset.experiment import DataView
from tests.helpers import TinyData
from utils.config import parse_args


def small_data(args,state=None):
    train=DataView(TinyData(20),range(20))
    val=DataView(TinyData(8),range(8))
    test=DataView(TinyData(8),range(8))
    return train,val,test,{'fixture':True}


class EntryTests(unittest.TestCase):
    def test_missing_checkpoint_fails_before_dataset_access(self):
        from main import main
        with patch('main.build_datasets',side_effect=AssertionError('dataset touched')):
            with self.assertRaises(FileNotFoundError):
                main(parse_args(['--test_checkpoint','missing-no-file.pth']))

    def test_train_evaluate_resume_and_reject_configuration_drift(self):
        from main import main
        with tempfile.TemporaryDirectory() as tmp, patch('main.build_datasets',side_effect=small_data), patch('main.cleanup_runtime'):
            output=Path(tmp)/'first'
            flags=['--dataset','PHISHING','--attack','LFBA','--anchor_idx','1','--poison_rate','.2',
                   '--select_rate','.5','--batch_size','4','--epoch','2','--device','-1','--early_stop','0',
                   '--results_dir',str(output)]
            first=main(parse_args(flags))
            self.assertTrue((output/'metrics.json').exists())
            self.assertFalse((output/'environment.json').exists())
            self.assertFalse((output/'best_checkpoint.pth.tar').exists())
            # Reconstruct an evaluation-only fixture from the retained last state.
            state=torch.load(output/'.runtime'/'last_checkpoint.pth.tar',weights_only=True)
            state.update(state_dict=state['best_state_dict'],epoch=state['best_epoch'],evaluation_only=True)
            best=Path(tmp)/'evaluation_fixture.pth.tar'
            torch.save(state,best)
            evaluated=main(parse_args(['--test_checkpoint',str(best),'--device','-1',
                                       '--results_dir',str(Path(tmp)/'eval')]))
            self.assertEqual(first['test'],evaluated['test'])
            last=output/'.runtime'/'last_checkpoint.pth.tar'
            last_state=torch.load(last,weights_only=True)
            last_eval=main(parse_args(['--test_checkpoint',str(last),'--device','-1',
                                       '--results_dir',str(Path(tmp)/'eval_last')]))
            self.assertEqual(last_eval['validation'],last_state['history'][-1]['validation'])
            self.assertEqual(last_eval['checkpoint_epoch'],last_state['epoch'])
            self.assertEqual(last_eval['best_epoch'],last_state['best_epoch'])
            main(parse_args(['--pretrained_checkpoint',str(last),'--epoch','3','--device','-1',
                             '--results_dir',str(Path(tmp)/'resumed')]))
            ck=torch.load(Path(tmp)/'resumed'/'.runtime'/'last_checkpoint.pth.tar',weights_only=True)
            self.assertEqual(ck['epoch'],3)
            with self.assertRaisesRegex(ValueError,'batch_size'):
                main(parse_args(['--pretrained_checkpoint',str(last),'--batch_size','8']))

    def test_legacy_evaluation_requires_explicit_metadata(self):
        from main import main, build_models
        with tempfile.TemporaryDirectory() as tmp, patch('main.build_datasets',side_effect=small_data):
            args=parse_args(['--dataset','PHISHING','--target_label','1'])
            models=build_models(args)
            ck=Path(tmp)/'legacy.pth'
            torch.save({'epoch':1,'state_dict':[m.state_dict() for m in models]},ck)
            with self.assertRaisesRegex(ValueError,'target_label'):
                main(parse_args(['--test_checkpoint',str(ck)]))
            with self.assertRaisesRegex(ValueError,'legacy|Legacy'):
                main(parse_args(['--pretrained_checkpoint',str(ck)]))
