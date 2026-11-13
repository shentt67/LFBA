"""Four-arm label-free selection; the legacy selector stays byte-identical."""
import time
import numpy as np

from attack.neighbor_vote import select_all, ordered_candidates

METHODS = ('auto_single', 'auto_rank_last', 'auto_mean', 'auto_vote')
SEEDS = [2004, 592, 3383, 100]


def schedule(epochs, count):
    if epochs <= 3 or count < 2:
        raise ValueError('Need more than three epochs and at least two candidates')
    return [0]*3 + [count//2]*(epochs-3)


def select_core(observations, count, block_size=128, device='cpu'):
    if [o['epoch'] for o in observations] != [1, 2, 3]:
        raise ValueError('Exactly three ordered clean observations required')
    started = time.perf_counter()
    selection = select_all(observations, count, block_size, device)
    ids = selection['ids'].numpy()
    anchor = selection['results']['auto_vote']['anchor_id']
    scores = selection['evidence']['auto_vote']['similarities'][-1].numpy()
    positions = ordered_candidates(ids, scores, count, anchor)
    selection['results']['auto_rank_last'] = dict(
        method='auto_rank_last', anchor_id=anchor,
        candidates=sorted(int(i) for i in ids[positions]),
        diagnostics=dict(valid_count=len(ids), excluded_count=len(observations[0]['ids'])-len(ids)))
    selection['results'] = {m: selection['results'][m] for m in METHODS}
    selection['elapsed_seconds'] = time.perf_counter()-started
    return selection


def diagnostics(selection):
    choices = selection['results']
    overlap = {}
    for a, b in [('auto_rank_last', 'auto_mean'), ('auto_rank_last', 'auto_vote'),
                 ('auto_mean', 'auto_vote')]:
        x, y = set(choices[a]['candidates']), set(choices[b]['candidates'])
        overlap[a+'__'+b] = len(x & y)/len(x | y)
    vote = choices['auto_vote']['diagnostics']
    return dict(jaccard=overlap, vote=vote,
                shared_selection_seconds=selection['elapsed_seconds'],
                valid_count=len(selection['ids']))
