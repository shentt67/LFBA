import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.trainer import Trainer
from dataset.experiment import DataView


class TinyData(Dataset):
    def __init__(self, n=20):
        self.data = torch.arange(n * 30, dtype=torch.float32).reshape(n, 30) / 100
        self.data_p = self.data.clone()
        self.targets = torch.arange(n) % 2
        self.ids = list(range(n))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = self.ids[index]
        return self.data[i], self.data_p[i], self.targets[i], i


class Bottom(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(15)
        self.fc = nn.Linear(15, 4)
        self.calls = []

    def forward(self, x):
        self.calls.append((self.training, torch.is_grad_enabled()))
        return self.fc(self.bn(x))


class Top(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(torch.cat(x, dim=1))


def make_trainer(directory, **overrides):
    Path(directory).mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(dataset='PHISHING', client_num=2, attack=None,
        pretrain_stage=0, start_epoch=0, epoch=2, lr=.001, print_steps=100,
        early_stop=20, target_label=1, results_dir=str(directory), batch_size=4,
        seed=100, poison_rate=.2, select_rate=.5, poison_all=False,
        random_select=False, attack_client_num=1, anchor_idx=1,
        similarity='legacy_l1', trigger_feature_clip=1.)
    vars(args).update(overrides)
    models = [Top(), Bottom(), Bottom()]
    opts = [torch.optim.Adam(m.parameters(), lr=args.lr) for m in models]
    generator = torch.Generator().manual_seed(101)
    loader = DataLoader(DataView(TinyData(),range(20)), batch_size=4, shuffle=True, generator=generator)
    ev = DataLoader(TinyData(8), batch_size=4, generator=torch.Generator().manual_seed(102))
    logger = logging.getLogger('lfba-tests')
    logger.addHandler(logging.NullHandler())
    return Trainer(torch.device('cpu'), models, [], None, opts, nn.CrossEntropyLoss(),
                   loader, ev, ev, [15], logger, args, val_loader=ev, val_asr_loader=ev)
