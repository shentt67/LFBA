import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from dataset.dataset import PHISHING_VFL
from dataset.utils import split_vfl
from types import SimpleNamespace
import torch


class DataTests(unittest.TestCase):
    def test_nuswide_names(self):
        for name in ['NUSWIDE','NUSWIDET']:
            a,b=split_vfl(torch.zeros(2,1634),SimpleNamespace(dataset=name))
            self.assertEqual((a.shape[1],b.shape[1]),(634,1000))
        with self.assertRaisesRegex(ValueError,'NUSWIDEI'):
            split_vfl(torch.zeros(2,1634),SimpleNamespace(dataset='NUSWIDEI'))

    def frame(self):
        frame=pd.DataFrame({f'f{i}':np.arange(200,dtype=float)+i for i in range(30)})
        frame['Result']=np.arange(200)%2
        return frame

    def test_phishing_disjoint_shared_splits_and_training_only_scaler(self):
        self.assertTrue(hasattr(PHISHING_VFL,'prepare'), 'prepare split and scaler once')
        frame=self.frame()
        with patch('dataset.dataset.pd.read_csv',return_value=frame):
            prepared=PHISHING_VFL.prepare('unused',seed=100)
            train=PHISHING_VFL('unused',True,None,prepared=prepared)
            val=PHISHING_VFL('unused',False,None,prepared=prepared,split='val')
            test=PHISHING_VFL('unused',False,None,prepared=prepared)
            again=PHISHING_VFL.prepare('unused',seed=100)
        for a,b in [(train,val),(train,test),(val,test)]:
            self.assertFalse(set(a.ids)&set(b.ids))
        self.assertEqual(prepared['state']['train_ids'],again['state']['train_ids'])
        expected=frame.drop(columns='Result').to_numpy()[train.ids].mean(axis=0)
        np.testing.assert_allclose(prepared['state']['mean'],expected)
        changed=frame.copy()
        changed.loc[test.ids,'f0']+=1e6
        with patch('dataset.dataset.pd.read_csv',return_value=changed):
            other=PHISHING_VFL.prepare('unused',seed=100)
        np.testing.assert_allclose(other['state']['mean'],expected)
        self.assertEqual(train[0][3],train.ids[0])

    def test_phishing_anchor_is_retained_before_scaler_fit(self):
        frame=self.frame()
        with patch('dataset.dataset.pd.read_csv',return_value=frame):
            initial=PHISHING_VFL.prepare('unused',seed=100)
            anchor=initial['state']['test_ids'][0]
            prepared=PHISHING_VFL.prepare('unused',seed=100,anchor_idx=anchor)
        self.assertIn(anchor,prepared['state']['train_ids'])
        self.assertNotIn(anchor,prepared['state']['test_ids'])
        expected=frame.drop(columns='Result').to_numpy()[prepared['state']['train_ids']].mean(0)
        np.testing.assert_allclose(prepared['state']['mean'],expected)
