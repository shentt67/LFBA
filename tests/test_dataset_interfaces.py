import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.experiment import build_datasets
from dataset.utils import split_vfl
from main import build_models
from utils.config import parse_args


class DatasetInterfaceTests(unittest.TestCase):
    def test_real_file_interfaces_and_attacker_boundaries_for_all_table_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rng=np.random.default_rng(42)
            (root/'Phishing').mkdir()
            frame=pd.DataFrame(rng.normal(size=(200,30)),columns=[f'f{i}' for i in range(30)])
            frame['Result']=np.arange(200)%2
            frame.to_csv(root/'Phishing'/'CM1.csv',index=False)
            for split,n in [('train',120),('test',60)]:
                folder=root/'UCI-HAR'/'UCI HAR Dataset'/split
                folder.mkdir(parents=True)
                np.savetxt(folder/f'X_{split}.txt',rng.normal(size=(n,561)))
                np.savetxt(folder/f'y_{split}.txt',np.arange(n)%6+1)
            nus=root/'NUS_WIDE'
            for folder in ['Groundtruth/TrainTestLabels','Low_Level_Features','NUS_WID_Tags']:
                (nus/folder).mkdir(parents=True)
            names=['buildings','grass','animal','water','person']
            for split,n in [('Train',100),('Test',50)]:
                for label,name in enumerate(names):
                    np.savetxt(nus/'Groundtruth'/'TrainTestLabels'/f'Labels_{name}_{split}.txt',
                               (np.arange(n)%5==label).astype(int),fmt='%d')
                np.savetxt(nus/'Low_Level_Features'/f'{split}_Normalized_features.dat',rng.normal(size=(n,634)))
                np.savetxt(nus/'NUS_WID_Tags'/f'{split}_Tags1k.dat',rng.normal(size=(n,1000)),delimiter='\t')
            for dataset,boundary in [('PHISHING',15),('UCIHAR',281),('NUSWIDE',634)]:
                with self.subTest(dataset=dataset):
                    args=parse_args(['--dataset',dataset,'--data_dir',tmp,'--target_label','1'])
                    train,val,test,state=build_datasets(args)
                    self.assertFalse(set(train.ids)&set(val.ids))
                    self.assertFalse(set(train.ids)&set(test.ids))
                    restored=build_datasets(args,state)
                    self.assertEqual(restored[0].ids,train.ids)
                    if dataset=='PHISHING':
                        args.eval_limit=10
                        with self.assertRaisesRegex(ValueError,'eval_limit'):
                            build_datasets(args)
                        args.eval_limit=None
                    recipient,donor=train.ids[:2]
                    train.set_attack({recipient:donor},args,[boundary])
                    x,clean,_,original=train[0]
                    torch.testing.assert_close(torch.as_tensor(x[:boundary]),torch.as_tensor(clean[:boundary]))
                    self.assertEqual(original,recipient)
                    models=build_models(args)
                    parts=split_vfl(torch.as_tensor(np.stack([train[i][0] for i in range(4)]),dtype=torch.float32),args)
                    logits=models[0]([m(p) for m,p in zip(models[1:],parts)])
                    self.assertEqual(logits.shape[0],4)
                    logits.sum().backward()
