"""Sweep k and pick the number of topics — the GPU-accelerated core.

Running k-means for every k in a wide range is exactly what flash-kmeans makes
feasible on large corpora. Two selection strategies: ``elbow`` (kneed on the inertia
curve, default) and ``silhouette`` (max mean cosine silhouette). The fit at the chosen
k is returned, so the caller doesn't refit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flashtopic.cluster.backends import KMeansResult, spherical_kmeans


@dataclass
class SweepResult:
    best_k: int
    ks: list[int]
    inertias: list[float]
    strategy: str
    backend: str
    result: KMeansResult  # the fit at best_k


def sampled_silhouette(
    X: np.ndarray, labels: np.ndarray, *, sample: int = 4000, seed: int = 0
) -> float:
    """Mean cosine silhouette on a random sample (no sklearn). X must be L2-normalized."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(sample, n), replace=False)
    Xs, ls = X[idx], np.asarray(labels)[idx]
    uniq = np.unique(ls)
    if len(uniq) < 2:
        return 0.0
    D = 1.0 - Xs @ Xs.T
    np.fill_diagonal(D, 0.0)
    sil = np.zeros(len(Xs))
    for i in range(len(Xs)):
        same = ls == ls[i]
        n_same = int(same.sum()) - 1
        if n_same <= 0:
            # a(i) is undefined for a lone member; 0 is the standard convention. Scoring
            # it as (b - 0)/b = 1 would make "every point its own cluster" the global
            # optimum, which silently rewards degenerately fine clusterings.
            continue
        a = D[i, same].sum() / n_same
        b = min(D[i, ls == c].mean() for c in uniq if c != ls[i])
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(sil.mean())


def sweep_k(
    X: np.ndarray,
    ks,
    *,
    backend: str = "auto",
    strategy: str = "elbow",
    seed: int = 0,
    n_iter: int = 50,
    n_init: int = 1,
    silhouette_sample: int = 4000,
) -> SweepResult:
    """Cluster `X` at each k in `ks`; pick the best by `strategy`.

    strategy: ``"elbow"`` (kneed on inertia) or ``"silhouette"``. ``n_init`` restarts each
    k-means and keeps the lowest-inertia fit. Returns the full inertia curve plus the fit
    at the chosen k.
    """
    ks = [int(k) for k in ks if 2 <= int(k) < X.shape[0]]
    if not ks:
        raise ValueError("no valid k values to sweep (need 2 <= k < n_points)")

    fits: dict[int, KMeansResult] = {}
    inertias: list[float] = []
    for k in ks:
        r = spherical_kmeans(X, k, backend=backend, n_iter=n_iter, seed=seed, n_init=n_init)
        fits[k] = r
        inertias.append(r.inertia)
    used_backend = fits[ks[0]].backend

    if len(ks) == 1:
        best_k = ks[0]
    elif strategy == "elbow":
        from kneed import KneeLocator

        loc = KneeLocator(ks, inertias, curve="convex", direction="decreasing", S=1.0)
        best_k = int(loc.knee) if loc.knee is not None else ks[len(ks) // 2]
    elif strategy == "silhouette":
        Xn = _l2(X)
        scores = [
            sampled_silhouette(Xn, fits[k].labels, sample=silhouette_sample, seed=seed)
            for k in ks
        ]
        best_k = ks[int(np.argmax(scores))]
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use 'elbow' or 'silhouette'")

    return SweepResult(
        best_k=best_k, ks=ks, inertias=inertias, strategy=strategy,
        backend=used_backend, result=fits[best_k],
    )


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
