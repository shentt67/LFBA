import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms import ToTensor

from dataset.dataset import CIFAR10_VFL


def tiny_cifar():
    data = CIFAR10_VFL.__new__(CIFAR10_VFL)
    data.data = np.stack([np.full((32,32,3), 20, np.uint8), np.full((32,32,3), 150, np.uint8)])
    data.data_p = data.data.copy()
    data.targets = [0,1]
    data.train = True
    data.transform = ToTensor()
    data.target_transform = None
    return data


class CifarTests(unittest.TestCase):
    def test_post_augmentation_attack_preserves_honest_half(self):
        data = tiny_cifar()
        self.assertTrue(hasattr(data, 'set_attack_plan'), 'CIFAR needs a deferred attack plan')
        before = data.data.copy()
        data.set_attack_plan({0:1})
        for crop in [(0,0), (8,8), (4,4)]:
            for flip in [False,True]:
                with self.subTest(crop=crop,flip=flip), patch.object(data,'_sample_augmentation',return_value=(*crop,flip)):
                    x, clean, label, index = data[0]
                    torch.testing.assert_close(x[:,:,:16],clean[:,:,:16],rtol=0,atol=0)
                    self.assertFalse(torch.equal(x[:,:,16:],clean[:,:,16:]))
                    self.assertEqual(float(x[0,31,31]),1.)
                    self.assertEqual((label,index),(0,0))
        np.testing.assert_array_equal(data.data,before)

    def test_trigger_only_and_label_replacement(self):
        data=tiny_cifar()
        self.assertTrue(hasattr(data,'set_attack_plan'))
        data.set_attack_plan({0:None}, {0:1})
        with patch.object(data,'_sample_augmentation',return_value=(4,4,True)):
            x,clean,label,_=data[0]
        self.assertEqual(label,1)
        torch.testing.assert_close(x[:,:,:16],clean[:,:,:16])
        self.assertEqual(int((x!=clean).sum()),27)
