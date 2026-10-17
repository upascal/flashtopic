"""Coverage-aware exemplar selection.

Showing an LLM only the strongest-activating members (the cluster *core*) produces
*false coherence*: a cluster that is secretly two topics looks unified because you
only sampled near its centroid. `coverage_exemplars` samples across the cluster's
spread instead — medoid anchor + farthest-point sampling (surfaces the periphery and
any seams) + a stratified fill across distance-to-centroid quantiles.
"""

from __future__ import annotations

import numpy as np


def coverage_exemplars(
    emb_subset: np.ndarray,
    member_global_idx: list[int],
    n: int,
    *,
    fps_frac: float = 0.6,
    seed: int = 0,
) -> list[int]:
    """Pick up to `n` members spanning the cluster's spread; returns global indices.

    `emb_subset` (m, d) are L2-normalized member embeddings; `member_global_idx`
    maps row → global index.
    """
    m = emb_subset.shape[0]
    if m <= n:
        return list(member_global_idx)

    centroid = emb_subset.mean(axis=0)
    cn = np.linalg.norm(centroid)
    if cn > 0:
        centroid = centroid / cn
    sims = emb_subset @ centroid

    selected: list[int] = []
    selected_set: set[int] = set()

    # 1. medoid anchor — closest to centroid
    medoid = int(np.argmax(sims))
    selected.append(medoid)
    selected_set.add(medoid)

    # 2. farthest-point sampling on cosine distance until ~fps_frac of budget
    n_fps = max(1, int(round(n * fps_frac)))
    min_dist = 1.0 - (emb_subset @ emb_subset[medoid])
    while len(selected) < n_fps:
        cand = int(np.argmax(min_dist))
        if cand in selected_set:
            break
        selected.append(cand)
        selected_set.add(cand)
        min_dist = np.minimum(min_dist, 1.0 - (emb_subset @ emb_subset[cand]))

    # 3. stratified fill by distance-to-centroid quantiles (4 bins)
    rng = np.random.default_rng(seed)
    remaining = n - len(selected)
    if remaining > 0:
        order = np.argsort(sims)  # ascending: far-from-centroid → near
        per_bin = max(1, remaining // 4)
        for b in np.array_split(order, 4):
            pool = [int(i) for i in b if int(i) not in selected_set]
            if not pool:
                continue
            take = min(per_bin, len(pool), n - len(selected))
            if take <= 0:
                break
            for p in rng.choice(pool, size=take, replace=False):
                selected.append(int(p))
                selected_set.add(int(p))

    return [int(member_global_idx[i]) for i in selected[:n]]


def typical_exemplars(
    emb_subset: np.ndarray,
    member_global_idx: list[int],
    n: int,
    *,
    dedup_tau: float = 0.95,
) -> list[int]:
    """Up to `n` members closest to the centroid, skipping near-duplicates; global indices.

    The complement of :func:`coverage_exemplars`: coverage shows a labeler the cluster's
    *range* (so it picks outliers on purpose), while these show what the cluster is
    *about* — the right material for a codebook's positive examples.
    """
    X = emb_subset / np.maximum(np.linalg.norm(emb_subset, axis=1, keepdims=True), 1e-12)
    sims = X @ _unit(X.mean(axis=0))
    picked: list[int] = []
    for i in np.argsort(-sims):
        if len(picked) >= n:
            break
        if picked and float(np.max(X[picked] @ X[i])) >= dedup_tau:
            continue
        picked.append(int(i))
    return [member_global_idx[i] for i in picked]


def cluster_coherence(emb_subset: np.ndarray) -> dict[str, float]:
    """How tightly members sit around their centroid (cosine; higher = tighter).

    ``median`` summarizes the cluster; ``p10`` exposes a loose tail (outliers or a
    second idea riding along). Singletons score 1.0.
    """
    X = emb_subset / np.maximum(np.linalg.norm(emb_subset, axis=1, keepdims=True), 1e-12)
    sims = X @ _unit(X.mean(axis=0))
    return {"median": float(np.median(sims)), "p10": float(np.percentile(sims, 10))}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
