"""Fixed-anchor controls using the exact common-valid cosine/voting rules."""
import time
import numpy as np

from attack.neighbor_vote import align_observations, ordered_candidates

METHODS = ('fixed_single', 'fixed_vote')
AUTO_METHODS = ('auto_rank_last', 'auto_vote')
SEEDS = [2004, 592, 3383]
ANCHOR = 23470


def select_fixed(observations, count, anchor=ANCHOR):
    if [o['epoch'] for o in observations] != [1, 2, 3]:
        raise ValueError('Exactly three ordered clean observations required')
    started = time.perf_counter()
    ids, gradients = align_observations(observations, count)
    ids = ids.numpy()
    positions = np.flatnonzero(ids == anchor)
    if len(positions) != 1:
        raise ValueError('Fixed anchor is absent or invalid in the common valid set')
    position = int(positions[0])
    scores = np.stack([(g @ g[position]).numpy() for g in gradients])
    votes = np.zeros(len(ids), dtype=np.int64)
    neighbors = []
    for row in scores:
        chosen = ordered_candidates(ids, row, count, anchor)
        votes[chosen] += 1
        neighbors.append(ids[chosen].tolist())
    results = {}
    for method in METHODS:
        is_vote = method == 'fixed_vote'
        chosen = ordered_candidates(ids, scores.mean(0) if is_vote else scores[-1],
                                    count, anchor, votes if is_vote else None)
        diag = dict(valid_count=len(ids), excluded_count=len(observations[0]['ids'])-len(ids))
        if is_vote:
            cutoff = int(votes[chosen[-1]])
            boundary = (votes == cutoff) & (ids != anchor)
            selected = int(boundary[chosen].sum())
            tied = int(boundary.sum())
            diag.update(cutoff_votes=cutoff, boundary_tied_count=tied,
                        selected_at_boundary=selected, tiebreak_required=0 < selected < tied,
                        selected_by_tiebreak=selected if 0 < selected < tied else 0,
                        vote_histogram={str(k):int(((votes==k)&(ids!=anchor)).sum()) for k in range(4)})
        results[method] = dict(method=method, anchor_id=int(anchor),
                               candidates=sorted(int(i) for i in ids[chosen]), diagnostics=diag)
    return dict(results=results, evidence=dict(ids=ids.tolist(), similarities=scores.tolist(),
                                              votes=votes.tolist(), neighbors=neighbors),
                elapsed_seconds=time.perf_counter()-started)
