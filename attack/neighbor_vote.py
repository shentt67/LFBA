"""Exact, label-free automatic anchor selection and temporal neighbor voting."""
import time

import numpy as np
import torch

METHODS = ('auto_single', 'auto_mean', 'auto_vote')


def align_observations(observations, count, return_norms=False):
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
    normalized = [g[valid] / g[valid].norm(dim=1, keepdim=True) for g in aligned]
    if return_norms:
        raw_norms = [g[valid].norm(dim=1) for g in aligned]
        return reference[valid], normalized, raw_norms
    return reference[valid], normalized


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


def reference_radius(valid_count, ref_rate=0.08):
    """Scale-decoupled structural discovery radius K_ref independent of poison budget P."""
    if valid_count < 3:
        raise ValueError('At least 3 valid samples required')
    return max(1, min(int(round(valid_count * ref_rate)), (valid_count - 1) // 2))


@torch.no_grad()
def local_margins(normalized, count, block_size=128, device='cpu', norms=None):
    """No N-by-N matrix is retained. Inputs must already be L2 normalized."""
    if block_size < 1 or count < 1 or len(normalized) < 2 * count + 1:
        raise ValueError('Invalid block size or neighborhood size')
    vectors = normalized.to(device=device, dtype=torch.float32)
    log_norms = (torch.log1p(norms.to(device=vectors.device, dtype=torch.float32))
                 if norms is not None else None)
    # Disable TF32 so the specified FP32 metric is not silently approximated.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        margins = []
        energies = []
        for start in range(0, len(vectors), block_size):
            stop = min(start + block_size, len(vectors))
            similarities = vectors[start:stop] @ vectors.T
            similarities[torch.arange(stop-start, device=vectors.device),
                         torch.arange(start, stop, device=vectors.device)] = -torch.inf
            if log_norms is None:
                nearest = similarities.topk(2 * count, dim=1, sorted=True).values
                margins.append((nearest[:, :count].mean(1) - nearest[:, count:].mean(1)).cpu())
            else:
                nearest, indices = similarities.topk(2 * count, dim=1, sorted=True)
                margins.append((nearest[:, :count].mean(1) - nearest[:, count:].mean(1)).cpu())
                nbr_energy = 0.5 * log_norms[start:stop] + 0.5 * log_norms[indices[:, :count]].mean(1)
                energies.append(nbr_energy.cpu())
        if log_norms is None:
            return torch.cat(margins).numpy()
        return torch.cat(margins).numpy(), torch.cat(energies).numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def ordered_candidates(ids, scores, count, excluded, votes=None):
    mask = np.asarray(ids) != excluded
    positions = np.flatnonzero(mask)
    keys = (np.asarray(ids)[positions], -np.asarray(scores)[positions])
    if votes is not None:
        keys += (-np.asarray(votes)[positions],)
    return positions[np.lexsort(keys)[:count]]


def _select_medoid_anchor(ids, gradients, base_rank, k_ref, top_m=32):
    """Select an interior medoid anchor with high cross-round neighborhood coherence."""
    m = min(top_m, len(ids))
    if m < 2:
        return int(np.argmax(base_rank))
    # Order top-M candidates by descending base_rank, breaking ties by ascending ID.
    cand_positions = np.lexsort((ids, -base_rank))[:m]
    coherences = np.empty(m, dtype=np.float64)
    pos_ratios = np.empty(m, dtype=np.float64)
    for idx, pos in enumerate(cand_positions):
        anchor_id = int(ids[pos])
        per_round = np.stack([(g @ g[pos]).numpy() for g in gradients])
        mean_sim = per_round.mean(0)
        min_sim = per_round.min(0)
        nbrs = ordered_candidates(ids, mean_sim, k_ref, anchor_id)
        coherences[idx] = float(mean_sim[nbrs].mean())
        pos_ratios[idx] = float((min_sim[nbrs] > 0.0).mean())
    # Filter out boundary outliers whose multi-round positive neighborhood ratio is below median.
    median_pos = float(np.median(pos_ratios))
    valid_mask = pos_ratios >= median_pos
    if int(valid_mask.sum()) >= 2:
        coh_ranks = np.zeros(m, dtype=np.float64)
        coh_ranks[valid_mask] = percentile_ranks(coherences[valid_mask])
    else:
        valid_mask = np.ones(m, dtype=bool)
        coh_ranks = percentile_ranks(coherences) if m >= 2 else np.ones(m, dtype=np.float64)
    combined = np.where(valid_mask, 0.70 * base_rank[cand_positions] + 0.30 * coh_ranks, -np.inf)
    best_idx = int(np.lexsort((ids[cand_positions], -combined))[0])
    return int(cand_positions[best_idx])


def _gate_candidates_and_donors(ids, anchor, per_round, votes, count, k_ref):
    """Purity-Preserving Core Gating for candidates and Negative-Similarity Filtering for donors."""
    mean = per_round.mean(0)
    min_sim = per_round.min(0)
    max_sim = per_round.max(0)
    all_ordered = ordered_candidates(ids, mean, len(ids) - 1, anchor, votes)
    raw_chosen = all_ordered[:count]

    core_mask = (votes >= 2) & (mean > 0.0) & (ids != anchor)
    if int(core_mask.sum()) >= 8:
        mu_core = float(mean[core_mask].mean())
        std_core = float(mean[core_mask].std())
        tau_gate = max(0.02, mu_core - 1.5 * std_core)
    else:
        tau_gate = 0.0

    # Over-budget regime (e.g., p=0.20 where count exceeds single-cluster capacity ~0.10*N):
    # strictly gate candidate pool to the multi-round positive core within k_ref.
    if count > int(0.105 * len(ids)):
        cap = min(count, k_ref)
        gated = [int(p) for p in raw_chosen[:cap]
                 if votes[p] >= 2 and mean[p] >= tau_gate and min_sim[p] > 0.0]
        min_required = min(cap, max(8, k_ref // 4))
        if len(gated) < min_required:
            positive = [int(p) for p in raw_chosen[:cap] if mean[p] > 0.0]
            gated = positive if len(positive) >= min_required else raw_chosen[:cap].tolist()
        chosen = np.asarray(gated, dtype=np.int64)
    elif count > k_ref:
        # Intermediate regime (e.g., p=0.10): reject unvoted negative-similarity outliers if any.
        gated = [int(p) for p in raw_chosen if not (votes[p] == 0 and mean[p] <= 0.0)]
        min_required = min(count, k_ref)
        chosen = np.asarray(gated if len(gated) >= min_required else raw_chosen.tolist(), dtype=np.int64)
    else:
        # Small/moderate budget regime (count <= k_ref): all candidates come from the k_ref vote core.
        chosen = raw_chosen

    chosen_set = set(int(i) for i in ids[chosen]) | {int(anchor)}
    non_cand_mask = np.asarray([int(i) not in chosen_set for i in ids], dtype=bool)
    non_cand_positions = np.flatnonzero(non_cand_mask)

    strict_donor_mask = non_cand_mask & (votes == 0) & (mean <= 0.0) & (max_sim <= 0.05)
    strict_positions = np.flatnonzero(strict_donor_mask)
    min_donors = max(len(chosen), len(non_cand_positions) // 3)
    if len(strict_positions) >= min_donors:
        donor_positions = strict_positions
    else:
        # Fallback: sort non-candidates by ascending (votes, mean_similarity, id)
        order = np.lexsort((ids[non_cand_positions], mean[non_cand_positions], votes[non_cand_positions]))
        fallback_count = min(len(non_cand_positions), max(len(chosen), len(non_cand_positions) // 2))
        donor_positions = non_cand_positions[order[:fallback_count]]

    donor_pool = sorted(int(i) for i in ids[donor_positions])
    return chosen, donor_pool, float(tau_gate)


@torch.no_grad()
def select_all(observations, count, block_size=128, device='cpu',
               stabilized=None, ref_rate=0.08, leverage_weight=0.20):
    """Only IDs and gradients enter this interface. Labels never enter selection."""
    started = time.perf_counter()
    ids_t, gradients, norms = align_observations(observations, count, return_norms=True)
    ids = ids_t.numpy()
    use_stabilized = (len(ids) > 500) if stabilized is None else bool(stabilized)

    if not use_stabilized:
        margins = np.stack([local_margins(g, count, block_size, device) for g in gradients])
        average_rank = np.mean([percentile_ranks(row) for row in margins], axis=0)
        anchors = {'auto_single': int(np.argmax(margins[-1])),
                   'auto_mean': int(np.argmax(average_rank)),
                   'auto_vote': int(np.argmax(average_rank))}
        vote_radius = count
    else:
        k_ref = reference_radius(len(ids), ref_rate=ref_rate)
        pairs = [local_margins(g, k_ref, block_size, device, norms=n)
                 for g, n in zip(gradients, norms)]
        margins = np.stack([p[0] for p in pairs])
        energies = np.stack([p[1] for p in pairs])
        dir_rank = np.mean([percentile_ranks(row) for row in margins], axis=0)
        lev_rank = np.mean([percentile_ranks(row) for row in energies], axis=0)
        average_rank = (1.0 - leverage_weight) * dir_rank + leverage_weight * lev_rank
        medoid_pos = _select_medoid_anchor(ids, gradients, average_rank, k_ref)
        anchors = {'auto_single': int(np.argmax(margins[-1])),
                   'auto_mean': int(np.argmax(average_rank)),
                   'auto_vote': medoid_pos}
        vote_radius = k_ref

    results = {}
    evidence = {}
    for method, position in anchors.items():
        anchor = int(ids[position])
        per_round = np.stack([(g @ g[position]).numpy() for g in gradients])
        mean = per_round.mean(0)
        votes = np.zeros(len(ids), dtype=np.int64)
        neighbors = []
        radius = vote_radius if (use_stabilized and method == 'auto_vote') else count
        for scores in per_round:
            positions = ordered_candidates(ids, scores, radius, anchor)
            votes[positions] += 1
            neighbors.append(torch.from_numpy(ids[positions].copy()))
        evidence[method] = dict(neighbors=neighbors, votes=torch.from_numpy(votes.copy()),
                                similarities=torch.from_numpy(per_round.copy()))
        scores = per_round[-1] if method == 'auto_single' else mean
        donor_pool = None
        tau_gate = None
        if use_stabilized and method == 'auto_vote':
            chosen, donor_pool, tau_gate = _gate_candidates_and_donors(
                ids, anchor, per_round, votes, count, vote_radius)
        else:
            chosen = ordered_candidates(ids, scores, count, anchor,
                                        votes if method == 'auto_vote' else None)
        cutoff = int(votes[chosen[-1]]) if method == 'auto_vote' else None
        boundary = (votes == cutoff) & (ids != anchor) if cutoff is not None else np.zeros(len(ids), bool)
        diagnostics = dict(
            valid_count=len(ids), excluded_count=len(observations[0]['ids'])-len(ids),
            cutoff_votes=cutoff, boundary_tied_count=int(boundary.sum()),
            selected_at_boundary=int(boundary[chosen].sum()),
            tiebreak_required=bool(0 < boundary[chosen].sum() < boundary.sum()),
            selected_by_tiebreak=int(boundary[chosen].sum()) if 0 < boundary[chosen].sum() < boundary.sum() else 0,
            vote_histogram={str(k): int(((votes == k) & (ids != anchor)).sum())
                            for k in range(len(gradients)+1)})
        if use_stabilized and method == 'auto_vote':
            diagnostics.update(
                stabilized=True,
                reference_radius=int(vote_radius),
                requested_count=int(count),
                core_gated_count=int(len(chosen)),
                tau_gate=tau_gate,
                donor_pool_count=len(donor_pool))
        entry = dict(method=method, anchor_id=anchor,
                     candidates=sorted(int(i) for i in ids[chosen]),
                     diagnostics=diagnostics)
        if donor_pool is not None:
            entry['donor_pool'] = donor_pool
        results[method] = entry
    return dict(results=results, evidence=evidence, ids=ids_t, margins=torch.from_numpy(margins),
                average_rank=torch.from_numpy(average_rank),
                elapsed_seconds=time.perf_counter()-started)
