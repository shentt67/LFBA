"""One reference neighborhood, one anchor, budget-independent full rankings."""
import time

import numpy as np
import torch

from attack.neighbor_vote import align_observations, local_margins, ordered_candidates, percentile_ranks
from lfba_comparison.common import digest

METHODS = ('decoupled_mean', 'decoupled_vote')


@torch.no_grad()
def build_ranking(observations, reference_count, block_size=128, device='cpu'):
    """Only sample IDs and three clean gradient observations inform selection."""
    if [row['epoch'] for row in observations] != [1, 2, 3]:
        raise ValueError('Exactly three ordered clean observations required')
    started = time.perf_counter()
    ids_t, gradients = align_observations(observations, reference_count)
    ids = ids_t.numpy()
    margins = np.stack([local_margins(g, reference_count, block_size, device) for g in gradients])
    average_rank = np.mean([percentile_ranks(row) for row in margins], axis=0)
    position = int(np.argmax(average_rank))
    anchor = int(ids[position])
    similarities = np.stack([(g @ g[position]).numpy() for g in gradients])
    mean = similarities.mean(0)
    votes = np.zeros(len(ids), dtype=np.int64)
    neighbors = []
    for scores in similarities:
        positions = ordered_candidates(ids, scores, reference_count, anchor)
        votes[positions] += 1
        neighbors.append(ids[positions].tolist())
    rankings = {
        method: ids[ordered_candidates(ids, mean, len(ids) - 1, anchor,
                                       votes if method == 'decoupled_vote' else None)].tolist()
        for method in METHODS
    }
    return dict(reference_count=int(reference_count), anchor_id=anchor, rankings=rankings,
                ids=ids_t, margins=torch.from_numpy(margins),
                average_rank=torch.from_numpy(average_rank),
                similarities=torch.from_numpy(similarities), votes=torch.from_numpy(votes),
                neighbors=neighbors, excluded_count=len(observations[0]['ids']) - len(ids),
                elapsed_seconds=time.perf_counter() - started)


def choose(ranking, method, count):
    """Cut an immutable ranking; P never affects the anchor or the vote radius."""
    if method not in METHODS or not isinstance(count, int) or not 1 <= count <= len(ranking['ids']) - 1:
        raise ValueError('Unknown method or insufficient valid candidate capacity')
    ordered = ranking['rankings'][method]
    ids = ranking['ids'].tolist()
    positions = {sample: index for index, sample in enumerate(ids)}
    mean = ranking['similarities'].numpy().mean(0)
    votes = ranking['votes'].numpy()
    chosen = ordered[:count]
    cutoff = int(votes[positions[chosen[-1]]])
    tied = [sample for sample in ordered if votes[positions[sample]] == cutoff]
    selected_tied = len(set(tied) & set(chosen))
    boundary_mean = float(mean[positions[chosen[-1]]])
    next_mean = float(mean[positions[ordered[count]]]) if count < len(ordered) else None
    return dict(method=method, anchor_id=ranking['anchor_id'], candidates=sorted(chosen),
                reference_count=ranking['reference_count'], ranking_digest=digest(ordered),
                diagnostics=dict(valid_count=len(ids), excluded_count=ranking['excluded_count'],
                    cutoff_votes=cutoff if method == 'decoupled_vote' else None,
                    boundary_tied_count=len(tied) if method == 'decoupled_vote' else None,
                    selected_at_boundary=selected_tied if method == 'decoupled_vote' else None,
                    boundary_mean_similarity=boundary_mean, next_mean_similarity=next_mean,
                    mean_similarity_gap=None if next_mean is None else boundary_mean - next_mean,
                    vote_histogram={str(k): sum(int(votes[positions[i]]) == k for i in ordered)
                                    for k in range(4)}))
