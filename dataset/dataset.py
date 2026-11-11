import copy
from random import sample

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as TF
from .utils import get_labeled_data, validate_nuswide_files
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
from typing import Any, Callable, Optional, Tuple


class CIFAR10_VFL(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        self.data_p = copy.deepcopy(self.data)
        self.attack_plan = {}
        self.label_overrides = {}

    def set_attack_plan(self, receivers, label_overrides=None):
        """Map original recipient IDs to donor IDs (None means trigger only)."""
        plan = {int(i): None if j is None else int(j) for i,j in receivers.items()}
        if any(i < 0 or i >= len(self.data) or (j is not None and not 0 <= j < len(self.data))
               for i,j in plan.items()):
            raise ValueError('CIFAR attack plan contains an invalid original sample ID')
        self.attack_plan = plan
        self.label_overrides = {int(i):int(y) for i,y in (label_overrides or {}).items()}

    def _sample_augmentation(self):
        if not self.train:
            return 4, 4, False
        return int(torch.randint(9, ()).item()), int(torch.randint(9, ()).item()), bool(torch.rand(()) < .5)

    @staticmethod
    def _augment(image, params):
        top, left, flip = params
        image = TF.crop(TF.pad(image, 4), top, left, 32, 32)
        return TF.hflip(image) if flip else image

    def __getitem__(self, index):
        params = self._sample_augmentation()
        clean = np.array(self._augment(Image.fromarray(self.data_p[index]), params), copy=True)
        current = clean.copy()
        plan = getattr(self, 'attack_plan', {})
        if index in plan:
            donor = plan[index]
            if donor is not None:
                donor_image = np.asarray(self._augment(Image.fromarray(self.data_p[donor]), params))
                current[:,16:,:] = donor_image[:,16:,:]
            current[-3:,-3:,:] = 0
            current[-3,-1,:] = 255
            current[-1,-3,:] = 255
            current[-2,-2,:] = 255
            current[-1,-1,:] = 255
        img, img_poisoned = Image.fromarray(current), Image.fromarray(clean)
        if self.transform is not None:
            img, img_poisoned = self.transform(img), self.transform(img_poisoned)
        target = getattr(self, 'label_overrides', {}).get(index, self.targets[index])
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_poisoned, target, index


class UCIHAR_VFL(Dataset):
    def __init__(self, root, train, transforms):
        if train:
            self.data = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/train/X_train.txt')
            self.data_p = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/train/X_train.txt')
            self.targets = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/train/y_train.txt') - 1
        else:
            self.data = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/test/X_test.txt')
            self.data_p = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/test/X_test.txt')
            self.targets = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/test/y_test.txt') - 1

    def __getitem__(self, index):
        x = self.data[index]
        x_poisoned = self.data_p[index]
        y = self.targets[index]
        return x, x_poisoned, y, index

    def __len__(self):
        return len(self.data)


class NUSWIDE_VFL(Dataset):
    def __init__(self, root, selected_labels, train, transforms):
        validate_nuswide_files(os.path.join(root, 'NUS_WIDE'), selected_labels)
        if train:
            X_image, X_text, Y = get_labeled_data(os.path.join(root, 'NUS_WIDE'), selected_labels, None, 'Train')
            self.data = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)
            self.data_p = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)
            self.targets = torch.tensor(Y)
        else:
            X_image, X_text, Y = get_labeled_data(os.path.join(root, 'NUS_WIDE'), selected_labels, None, 'Test')
            self.data = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)
            self.data_p = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)
            self.targets = torch.tensor(Y)

    def __getitem__(self, index):
        x = self.data[index]
        x_poisoned = self.data_p[index]
        y = self.targets[index]
        return x, x_poisoned, y, index

    def __len__(self):
        return len(self.data)


class PHISHING_VFL(Dataset):
    @staticmethod
    def prepare(root, seed=100, validation_rate=.1, state=None, anchor_idx=None):
        from sklearn.model_selection import train_test_split
        frame = pd.read_csv(os.path.join(root, 'Phishing', 'CM1.csv'))
        features = frame.drop(columns=['Result']).to_numpy(dtype=np.float64)
        classes, labels = np.unique(frame['Result'].to_numpy(), return_inverse=True)
        if state is None:
            candidate, test = train_test_split(np.arange(len(labels)), test_size=.2,
                                              random_state=seed, stratify=labels)
            if anchor_idx is not None:
                if not 0 <= anchor_idx < len(labels):
                    raise ValueError('anchor_idx is outside PHISHING data')
                if anchor_idx in test:
                    replacement=min(int(i) for i in candidate if labels[i]==labels[anchor_idx])
                    candidate[candidate==replacement]=anchor_idx
                    test[test==anchor_idx]=replacement
            from .experiment import split_training_ids
            train, val = split_training_ids(labels,validation_rate,seed,anchor_idx,candidate)
            scaler = StandardScaler().fit(features[train])
            state = dict(train_ids=train, val_ids=val,
                         test_ids=sorted(test.tolist()), mean=scaler.mean_.tolist(),
                         scale=scaler.scale_.tolist(), classes=classes.tolist())
        scaled = (features - np.asarray(state['mean'])) / np.asarray(state['scale'])
        return dict(data=torch.tensor(scaled, dtype=torch.float32),
                    targets=torch.tensor(labels, dtype=torch.long), state=copy.deepcopy(state))

    def __init__(self, root, train, transforms=None, prepared=None, split=None, seed=100):
        prepared = prepared if prepared is not None else self.prepare(root, seed=seed)
        self.data = prepared['data']
        self.data_p = self.data.clone()
        self.targets = prepared['targets']
        self.state = prepared['state']
        self.ids = list(self.state[(split or ('train' if train else 'test')) + '_ids'])

    def __getitem__(self, index):
        i = self.ids[index]
        return self.data[i], self.data_p[i], self.targets[i], i

    def __len__(self):
        return len(self.ids)
