"""Exact, label-free automatic anchor selection and temporal neighbor voting."""
import time

import numpy as np
import torch

METHODS = ('auto_single', 'auto_mean', 'auto_vote')


def align_observations(observations, count):
    if not observations or count < 1:
        raise ValueError('Observations and a positive candidate count are required')
    reference = torch.as_tensor(observations[0]['ids'], dtype=torch.long).cpu().sort().values
    aligned = []
    valid = torch.ones(len(reference), dtype=torch.bool)
    for item in observations:
        ids = torch.as_tensor(item['ids'], dtype=torch.long).cpu()
        g = item['gradients'].detach().to(device='cpu', dtype=torch.float32)
        order = ids.argsort()
        if (len(ids.unique()) != len(ids) or not torch.equal(ids[order], reference)
                or g.ndim != 2 or len(g) != len(ids)):
            raise ValueError('Observation IDs must be unique and identical across rounds')
        g = g[order]
        norms = torch.linalg.vector_norm(g, dim=1)
        valid &= torch.isfinite(g).all(1) & torch.isfinite(norms) & (norms > 0)
        aligned.append(g)
    if int(valid.sum()) < 2 * count + 1:
        raise ValueError('Fewer than 2P+1 common valid samples')
    return reference[valid], [g[valid] / g[valid].norm(dim=1, keepdim=True) for g in aligned]


def percentile_ranks(values):
    """Ascending average ranks, mapped onto [0, 1]; exact ties share a rank."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Finite rank values required')
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    result = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        result[order[start:end]] = (start + end - 1) / (2 * (len(values) - 1))
    return result


@torch.no_grad()
def local_margins(normalized, count, block_size=128, device='cpu'):
    """No N-by-N matrix is retained. Inputs must already be L2 normalized."""
    if block_size < 1 or count < 1 or len(normalized) < 2 * count + 1:
        raise ValueError('Invalid block size or neighborhood size')
    vectors = normalized.to(device=device, dtype=torch.float32)
    # Disable TF32 so the specified FP32 metric is not silently approximated.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        margins = []
        for start in range(0, len(vectors), block_size):
            stop = min(start + block_size, len(vectors))
            similarities = vectors[start:stop] @ vectors.T
            similarities[torch.arange(stop-start, device=vectors.device),
                         torch.arange(start, stop, device=vectors.device)] = -torch.inf
            nearest = similarities.topk(2 * count, dim=1, sorted=True).values
            margins.append((nearest[:, :count].mean(1) - nearest[:, count:].mean(1)).cpu())
        return torch.cat(margins).numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def ordered_candidates(ids, scores, count, excluded, votes=None):
    mask = np.asarray(ids) != excluded
    positions = np.flatnonzero(mask)
    keys = (np.asarray(ids)[positions], -np.asarray(scores)[positions])
    if votes is not None:
        keys += (-np.asarray(votes)[positions],)
    return positions[np.lexsort(keys)[:count]]


@torch.no_grad()
def select_all(observations, count, block_size=128, device='cpu'):
    """Only IDs and gradients enter this interface. Labels never enter selection."""
    started = time.perf_counter()
    ids_t, gradients = align_observations(observations, count)
    ids = ids_t.numpy()
    margins = np.stack([local_margins(g, count, block_size, device) for g in gradients])
    average_rank = np.mean([percentile_ranks(row) for row in margins], axis=0)
    # IDs are ascending, so argmax implements the declared ID tie break.
    anchors = {'auto_single': int(np.argmax(margins[-1])),
               'auto_mean': int(np.argmax(average_rank)),
               'auto_vote': int(np.argmax(average_rank))}
    results = {}
    evidence = {}
    for method, position in anchors.items():
        anchor = int(ids[position])
        per_round = np.stack([(g @ g[position]).numpy() for g in gradients])
        mean = per_round.mean(0)
        votes = np.zeros(len(ids), dtype=np.int64)
        neighbors = []
        for scores in per_round:
            positions = ordered_candidates(ids, scores, count, anchor)
            votes[positions] += 1
            neighbors.append(torch.from_numpy(ids[positions].copy()))
        evidence[method] = dict(neighbors=neighbors, votes=torch.from_numpy(votes.copy()),
                                similarities=torch.from_numpy(per_round.copy()))
        scores = per_round[-1] if method == 'auto_single' else mean
        chosen = ordered_candidates(ids, scores, count, anchor,
                                    votes if method == 'auto_vote' else None)
        cutoff = int(votes[chosen[-1]]) if method == 'auto_vote' else None
        boundary = (votes == cutoff) & (ids != anchor) if cutoff is not None else np.zeros(len(ids), bool)
        results[method] = dict(method=method, anchor_id=anchor,
            candidates=sorted(int(i) for i in ids[chosen]),
            diagnostics=dict(valid_count=len(ids), excluded_count=len(observations[0]['ids'])-len(ids),
                cutoff_votes=cutoff, boundary_tied_count=int(boundary.sum()),
                selected_at_boundary=int(boundary[chosen].sum()),
                tiebreak_required=bool(0 < boundary[chosen].sum() < boundary.sum()),
                selected_by_tiebreak=int(boundary[chosen].sum()) if 0 < boundary[chosen].sum() < boundary.sum() else 0,
                vote_histogram={str(k): int(((votes == k) & (ids != anchor)).sum())
                                for k in range(len(gradients)+1)}))
    return dict(results=results, evidence=evidence, ids=ids_t, margins=torch.from_numpy(margins),
                average_rank=torch.from_numpy(average_rank),
                elapsed_seconds=time.perf_counter()-started)
