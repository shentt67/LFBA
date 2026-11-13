"""Three-round shared-prefix training isolated from legacy two-arm behavior."""
import time
import numpy as np
import torch

from lfba_comparison.trainer import ComparisonTrainer
from lfba_comparison.common import digest
from lfba_comparison.core_policy import schedule
from utils.checkpoint import cpu_tree, capture_rng
from utils.trainer import Trainer


class CoreTrainer(ComparisonTrainer):
    def __init__(self, *args, choice=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.clean_window = 3
        self.pending_choice = cpu_tree(choice)

    def _restore(self):
        # Deep copies are essential: Adam load_state_dict may alias CPU step tensors.
        if self.checkpoint:
            self.checkpoint = cpu_tree(self.checkpoint)
        Trainer._restore(self)
        if self.checkpoint:
            state = self.checkpoint['comparison']
            source = state['method']
            branching = source == 'capture' and self.method != 'capture'
            if source != self.method and not (branching and self.checkpoint['epoch'] == 3):
                raise ValueError('Only a complete clean three-round prefix may branch')
            for name in ('choice', 'observations', 'audit', 'plans', 'prefix', 'cumulative',
                         'initial_digest', 'start_state', 'selection_seconds', 'shared_clean_seconds'):
                setattr(self, name, cpu_tree(state[name]))
            self.donor_rng.setstate(state['donor_rng'])
            self.elapsed_prior = state['elapsed_seconds']
            if branching:
                if self.choice or self.current_plan or self.plans or self.cumulative:
                    raise ValueError('Contaminated clean prefix')
                actual = self.state_identity()
                expected = dict(model=self.checkpoint['state_dict'],
                                optimizer=self.checkpoint['optimizer'], rng=self.checkpoint['rng_state'])
                if digest(actual) != digest(expected):
                    raise ValueError('Three-round restored state differs')
                self.start_state = digest(actual)
                self.elapsed_prior = 0.
                self.shared_clean_seconds = state['elapsed_seconds']
            if self.choice is not None and self.pending_choice != self.choice:
                raise ValueError('Saved choice differs from frozen selection')
        self.started = time.perf_counter()
        self._bind_target()

    def state_identity(self):
        return dict(model=[m.state_dict() for m in self.model_list],
                    optimizer=[o.state_dict() for o in self.optimizer_list],
                    rng=capture_rng(self._loaders()))

    def _configure_attack(self, ep):
        self.epoch_started = time.perf_counter()
        self.epoch_selection_seconds = 0.
        if self.method == 'capture' or ep < 3:
            if self.current_plan: raise ValueError('Poisoning in clean prefix')
            return
        if self.choice is None:
            if [o['epoch'] for o in self.observations] != [1, 2, 3] or self.pending_choice is None:
                raise ValueError('Missing frozen selection or clean observations')
            self.choice = cpu_tree(self.pending_choice)
            self._bind_target()
        candidates = self.choice['candidates']
        count = int(len(self.train_loader.dataset)*self.args.poison_rate)
        if len(set(candidates)) != count or self.choice['anchor_id'] in candidates:
            raise ValueError('Invalid candidate set')
        if self.poison_indexes is None: self.poison_indexes = list(candidates)
        if self.poison_indexes != candidates: raise ValueError('Frozen candidates changed')
        ids = self.indexes_epoch
        positions = torch.nonzero(torch.isin(ids, torch.tensor(candidates))).flatten()
        if len(positions) != count: raise ValueError('Missing candidate IDs')
        norms = self.grad_vec_epoch[positions].norm(dim=1)
        if not torch.isfinite(norms).all(): raise ValueError('Nonfinite hard-sample gradient')
        order = np.lexsort((ids[positions].numpy(), -norms.numpy()))
        q = schedule(self.args.epoch, count)[ep]
        recipients = ids[positions[torch.as_tensor(order[:q].copy())]].tolist()
        donors = self.donor_rng.sample(sorted(set(self.train_loader.dataset.ids)-set(candidates)), q)
        self.current_plan = dict(zip(recipients, donors))
        self.label_overrides = {}
        self._apply_attack()
        self.cumulative += q
        self.plans.append(dict(epoch=ep+1, recipients=recipients, donors=donors))
