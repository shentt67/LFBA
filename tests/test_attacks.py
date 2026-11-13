import logging
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from attack.attack import attack_rsa, get_near_index


class AttackTests(unittest.TestCase):
    def test_similarity_modes_zero_vectors_and_original_id_ties(self):
        anchor=torch.tensor([1.,0.,0.])
        x=torch.tensor([[1.,1.,1.],[1.,1.9,0.],[0.,0.,0.],[float('nan'),1.,0.]])
        self.assertEqual(get_near_index(anchor,x,1).tolist(),[1])
        self.assertEqual(get_near_index(anchor,x,1,mode='cosine').tolist(),[0])
        same=torch.tensor([[1.,0.],[1.,0.]])
        self.assertEqual(get_near_index(same[0],same,1,sample_ids=torch.tensor([8,3])).tolist(),[1])
        with self.assertRaisesRegex(ValueError,'valid'):
            get_near_index(anchor,x,3)
        with self.assertRaisesRegex(ValueError,'anchor'):
            get_near_index(torch.zeros(3),x,1)

    def test_rsa_preserves_numpy_and_tensor_inputs(self):
        for dataset, data, dims in [
            ('CIFAR10', np.zeros((4,32,32,3), dtype=np.uint8), []),
            ('PHISHING', torch.zeros(4,30), [15]),
        ]:
            with self.subTest(dataset=dataset):
                before = data.copy() if isinstance(data,np.ndarray) else data.clone()
                result = attack_rsa(SimpleNamespace(dataset=dataset), logging.getLogger(), data, dims, .5, 'test')
                np.testing.assert_array_equal(data, before)
                self.assertEqual(int(np.any(np.asarray(result) != np.asarray(before), axis=tuple(range(1,result.ndim))).sum()), 2)
