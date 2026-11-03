"""Labels do not enter either candidate selection policy."""
import torch

from attack.attack import get_near_index

METHODS = ('lfba_original', 'lfba_matched', 'auto_vote')
ANCHOR = 23470


def schedule(method, epochs, candidate_count):
    if method not in METHODS or epochs <= 3 or candidate_count < 2:
        raise ValueError('Invalid method, horizon or candidate count')
    per_epoch = candidate_count // 2
    if method == 'auto_vote': return [0]*3 + [per_epoch]*(epochs-3)
    if method == 'lfba_original': return [0] + [per_epoch]*(epochs-1)
    base, remainder = divmod((epochs-3)*per_epoch, epochs-1)
    return [0] + [base + int(i < remainder) for i in range(epochs-1)]


def fixed_choice(observations, method, count, anchor=ANCHOR):
    if method not in ('lfba_original','lfba_matched'):
        raise ValueError('Not a fixed-anchor method')
    if not observations or observations[0]['epoch'] != 1:
        raise ValueError('The first clean observation is required')
    first = observations[0]
    ids, gradients = first['ids'], first['gradients']
    position = torch.nonzero(ids == anchor).flatten()
    if len(position) != 1: raise ValueError('Predeclared anchor is missing')
    positions = get_near_index(gradients[int(position.item())], gradients, count, 'legacy_l1', ids)
    return dict(method=method, anchor_id=anchor, candidates=sorted(ids[positions].tolist()))


def purity(labels, candidates, anchor):
    target = int(labels[anchor])
    without = [i for i in candidates if i != anchor]
    return dict(full=sum(int(labels[i]) == target for i in candidates)/len(candidates),
                full_denominator=len(candidates),
                without_anchor=(sum(int(labels[i]) == target for i in without)/len(without)) if without else None,
                without_anchor_denominator=len(without), anchor_in_pool=anchor in candidates)
