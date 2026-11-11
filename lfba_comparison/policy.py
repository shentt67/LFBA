"""Labels do not enter either candidate selection policy."""
import torch

from attack.attack import get_near_index

METHODS = ('lfba_original', 'auto_vote')
ANCHOR = 2370
ANCHORS = {'CIFAR10': ANCHOR, 'UCIHAR': 27, 'NUSWIDE': 3390}


def schedule(method, epochs, candidate_count, select_rate=.5):
    if method not in METHODS or epochs <= 3 or candidate_count < 2:
        raise ValueError('Invalid method, horizon or candidate count')
    if not 0 < select_rate <= 1:
        raise ValueError('Select rate must be within (0,1]')
    per_epoch = int(candidate_count * select_rate)
    if per_epoch < 1: raise ValueError('Select rate yields zero replacements per epoch')
    if method == 'auto_vote': return [0]*3 + [per_epoch]*(epochs-3)
    return [0] + [per_epoch]*(epochs-1)


def fixed_choice(observations, method, count, anchor=ANCHOR):
    if method != 'lfba_original':
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
