"""Shared repaired training with method-specific online construction and budgets."""
from collections import Counter
import random
import time

import numpy as np
import torch

from attack.neighbor_vote import select_all
from dataset.utils import split_vfl
from utils.checkpoint import capture_rng, cpu_tree
from utils.trainer import Trainer
from lfba_comparison.common import digest
from lfba_comparison.policy import fixed_choice, purity, schedule


def _recipient_order(ids_subset, grad_subset, plans, stabilized=False):
    """Rank candidate positions for replacement using direction gating and exposure decay when stabilized."""
    norms = grad_subset.norm(dim=1)
    if not torch.isfinite(norms).all():
        raise ValueError('Nonfinite hard-sample gradient')
    if not stabilized:
        return np.lexsort((ids_subset.numpy(), -norms.numpy()))
    unit_grads = grad_subset / norms.unsqueeze(1).clamp_min(1e-12)
    mean_dir = unit_grads.mean(dim=0)
    mean_norm = float(mean_dir.norm().item())
    if mean_norm > 1e-12:
        cos_to_pool = (unit_grads @ (mean_dir / mean_norm)).cpu().numpy()
    else:
        cos_to_pool = np.ones(len(ids_subset), dtype=np.float64)
    # Candidates aligned with the candidate-pool mean direction get normal weight;
    # opposing directions (cos <= 0, e.g. wrong-class outliers) are strongly attenuated.
    dir_weight = np.where(
        cos_to_pool > 0.0,
        0.5 + 0.5 * cos_to_pool,
        0.05 * np.exp(2.0 * np.clip(cos_to_pool, -1.0, 0.0)),
    )
    used = Counter(int(r) for plan in plans for r in plan.get('recipients', []))
    exposures = np.asarray([used.get(int(i), 0) for i in ids_subset.tolist()], dtype=np.float64)
    priority = (norms.cpu().numpy() * dir_weight) / np.power(1.0 + exposures, 0.75)
    return np.lexsort((ids_subset.numpy(), -priority))


class ComparisonTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = self.args.comparison_method
        self.clean_window = 3 if self.method == 'auto_vote' else 1
        self.choice = None
        self.observations = []
        self.audit = []
        self.plans = []
        self.prefix = []
        self.cumulative = 0
        self.donor_rng = random.Random(self.args.seed+17000)
        self.initial_digest = digest([m.state_dict() for m in self.model_list])
        self.start_state = None
        self.elapsed_prior = 0.
        self.selection_seconds = 0.
        self.shared_clean_seconds = 0.
        self.started = time.perf_counter()

    def adjust_learning_rate(self, epoch):
        for optimizer in self.optimizer_list:
            for group in optimizer.param_groups: group['lr'] = .001

    def _restore(self):
        if self.checkpoint:
            # Adam can retain CPU state tensors (notably step) from load_state_dict.
            # Detach them before loading so this branch cannot mutate the shared
            # clean-prefix reference used by later groups and final audits.
            self.checkpoint = dict(self.checkpoint)
            self.checkpoint['optimizer'] = cpu_tree(self.checkpoint['optimizer'])
        super()._restore()
        if self.checkpoint:
            state = self.checkpoint['comparison']
            source = state['method']
            if source != self.method and not (source == 'capture' and self.checkpoint['epoch'] == 1):
                raise ValueError('Cannot resume a different method or non-prefix checkpoint')
            for name in ('choice','observations','audit','plans','prefix','cumulative','initial_digest',
                         'start_state','selection_seconds','shared_clean_seconds'):
                setattr(self,name,cpu_tree(state[name]))
            self.donor_rng.setstate(state['donor_rng'])
            self.elapsed_prior = state['elapsed_seconds']
            if source == 'capture' and self.method != 'capture':
                actual = dict(model=[m.state_dict() for m in self.model_list],
                    optimizer=[o.state_dict() for o in self.optimizer_list],rng=capture_rng(self._loaders()))
                expected = dict(model=self.checkpoint['state_dict'],optimizer=self.checkpoint['optimizer'],
                                rng=self.checkpoint['rng_state'])
                if digest(actual) != digest(expected): raise ValueError('First-round state restore differs')
                self.start_state = digest(actual)
                self.elapsed_prior = 0.
                self.shared_clean_seconds = state['elapsed_seconds']
        self.started = time.perf_counter()
        self._bind_target()

    def _bind_target(self):
        if self.choice is not None:
            self.args.anchor_idx = self.choice['anchor_id']
            # Evaluation metadata is joined only after label-free candidate selection.
            self.args.target_label = int(self.train_loader.dataset.targets[self.args.anchor_idx])

    def _snapshot(self):
        snapshot = super()._snapshot()
        snapshot['comparison'] = cpu_tree(dict(method=self.method,choice=self.choice,
            observations=self.observations,audit=self.audit,plans=self.plans,prefix=self.prefix,
            cumulative=self.cumulative,initial_digest=self.initial_digest,start_state=self.start_state,
            donor_rng=self.donor_rng.getstate(),selection_seconds=self.selection_seconds,
            shared_clean_seconds=self.shared_clean_seconds,
            elapsed_seconds=self.elapsed_prior+time.perf_counter()-self.started))
        return snapshot

    def _configure_attack(self, ep):
        self.epoch_started = time.perf_counter()
        self.epoch_selection_seconds = 0.
        if self.method == 'capture' or ep < self.clean_window:
            if self.current_plan: raise ValueError('Poisoning in clean prefix')
            return
        requested_count = int(len(self.train_loader.dataset)*self.args.poison_rate)
        if self.choice is None:
            if [o['epoch'] for o in self.observations] != list(range(1,self.clean_window+1)):
                raise ValueError('Missing required online clean observations')
            started = time.perf_counter()
            if self.method == 'auto_vote':
                selection = select_all(self.observations,requested_count,self.args.block_size,str(self.device))
                self.choice = selection['results']['auto_vote']
            else:
                self.choice = fixed_choice(self.observations,self.method,requested_count,self.args.anchor_idx)
            self.epoch_selection_seconds = time.perf_counter()-started
            self.selection_seconds += self.epoch_selection_seconds
            self._bind_target()
        candidates = self.choice['candidates']
        candidate_count = len(candidates)
        if self.poison_indexes is None: self.poison_indexes = list(candidates)
        if self.poison_indexes != candidates: raise ValueError('Frozen candidates changed')
        ids = self.indexes_epoch
        positions = torch.nonzero(torch.isin(ids,torch.tensor(candidates))).flatten()
        if len(positions) != candidate_count: raise ValueError('Candidate IDs missing')
        stabilized = bool(self.method == 'auto_vote' and self.choice.get('diagnostics', {}).get('stabilized'))
        order = _recipient_order(ids[positions], self.grad_vec_epoch[positions], self.plans, stabilized=stabilized)
        replacements = schedule(self.method,self.args.epoch,candidate_count,self.args.select_rate)[ep]
        recipients = ids[positions[torch.as_tensor(order[:replacements].copy())]].tolist()
        valid_donors = set(self.train_loader.dataset.ids)-set(candidates)
        filtered_pool = (sorted(set(self.choice.get('donor_pool') or ()) & valid_donors)
                         if self.method == 'auto_vote' else [])
        donor_source = filtered_pool if len(filtered_pool) >= replacements else sorted(valid_donors)
        donors = self.donor_rng.sample(donor_source,replacements)
        self.current_plan = dict(zip(recipients,donors))
        self.label_overrides = {}
        self._apply_attack()
        self.cumulative += replacements
        self.plans.append(dict(epoch=ep+1,recipients=recipients,donors=donors))

    def _clean_validation(self):
        correct = total = 0
        loss = 0.
        for model in self.model_list: model.eval()
        with torch.no_grad():
            for x,_,y,_ in self.val_loader:
                x=x.to(self.device,dtype=torch.float32);y=y.to(self.device,dtype=torch.long)
                output=self.model_list[0]([m(p) for m,p in zip(self.model_list[1:],split_vfl(x,self.args))])
                total+=len(y);correct+=int((output.argmax(1)==y).sum())
                loss+=float(self.criterion(output,y))*len(y)
        return dict(acc=correct/total,loss=loss/total,asr=None,poison_acc=None,target_acc=None,
                    counts=dict(acc=total),undefined={'asr':'clean observation prefix'})

    def test(self,ep=0,split='test'):
        if split != 'val': return super().test(ep,split)
        if self.method == 'capture' or ep < self.clean_window:
            if self.current_plan: raise ValueError('Contaminated gradient observation')
            order=self.indexes_epoch.argsort()
            self.observations.append(dict(epoch=ep+1,ids=self.indexes_epoch[order].clone(),
                                          gradients=self.grad_vec_epoch[order].clone()))
            self.prefix.append(dict(epoch=ep+1,model=digest([m.state_dict() for m in self.model_list]),
                optimizer=digest([o.state_dict() for o in self.optimizer_list]),rng=digest(capture_rng(self._loaders()))))
            metrics=self._clean_validation()
        else:
            metrics=super().test(ep,split)
        candidates=[] if self.choice is None else self.choice['candidates']
        recipients=list(self.current_plan)
        labels=np.asarray(self.train_loader.dataset.targets)
        p=purity(labels,candidates,self.choice['anchor_id']) if candidates else None
        self.audit.append(dict(epoch=ep+1,candidate_count=len(candidates),candidate_digest=digest(candidates),
            candidate_purity=None if p is None else p['full'],
            candidate_purity_without_anchor=None if p is None else p['without_anchor'],
            candidate_denominator=0 if p is None else p['full_denominator'],
            candidate_without_anchor_denominator=0 if p is None else p['without_anchor_denominator'],
            recipient_count=len(recipients),recipient_purity=float(np.mean(labels[recipients]==self.args.target_label)) if recipients else None,
            cumulative_replacements=self.cumulative,validation=metrics,
            epoch_seconds=time.perf_counter()-self.epoch_started,selection_seconds=self.epoch_selection_seconds))
        return metrics
