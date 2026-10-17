"""Cluster backend + k-sweep tests.

Backend-agnostic: runs against whatever backend is installed (auto), and skips the
clustering assertions entirely if none is present. The detection logic itself is
pure and always tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from flashtopic.cluster import (
    BACKENDS,
    available_backends,
    detect_backend,
    spherical_kmeans,
    sweep_k,
)

try:
    from sklearn.metrics import adjusted_rand_score
except Exception:  # pragma: no cover
    adjusted_rand_score = None

_HAVE_BACKEND = bool(available_backends())
_needs_backend = pytest.mark.skipif(not _HAVE_BACKEND, reason="no clustering backend installed")


def _blobs(k_true=6, d=32, per=300, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k_true, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X, y = [], []
    for c in range(k_true):
        X.append(centers[c] + spread * rng.normal(size=(per, d)))
        y += [c] * per
    X = np.vstack(X).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, np.array(y)


def test_detection_is_capability_based():
    avail = available_backends()
    assert all(b in BACKENDS for b in avail)
    if avail:
        assert detect_backend() == avail[0]  # priority order
    else:
        with pytest.raises(RuntimeError):
            detect_backend()


@_needs_backend
def test_spherical_kmeans_recovers_blobs():
    X, y = _blobs()
    r = spherical_kmeans(X, 6, seed=0)
    assert r.centroids.shape == (6, X.shape[1])
    # centroids are L2-normalized
    np.testing.assert_allclose(np.linalg.norm(r.centroids, axis=1), 1.0, atol=1e-4)
    if adjusted_rand_score is not None:
        assert adjusted_rand_score(y, r.labels) > 0.9


@_needs_backend
def test_sweep_elbow_finds_true_k():
    X, _ = _blobs(k_true=6)
    sw = sweep_k(X, range(2, 12), strategy="elbow", seed=0)
    assert sw.best_k in (5, 6, 7)  # elbow lands at/near the true count
    assert sw.inertias == sorted(sw.inertias, reverse=True)  # inertia decreases in k


@_needs_backend
def test_trivial_when_points_leq_k():
    X = np.eye(4, dtype=np.float32)
    r = spherical_kmeans(X, 10, seed=0)
    assert len(r.labels) == 4 and len(set(r.labels.tolist())) == 4


def test_silhouette_does_not_reward_all_singletons():
    """Backend-free regression: a lone cluster member scores 0, not 1.

    a(i) is undefined for a singleton. Scoring it as (b - 0)/b = 1 made "every point in
    its own cluster" the global silhouette optimum, which silently drove k-selection to
    the finest resolution available.
    """
    import numpy as np

    from flashtopic.cluster import sampled_silhouette

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    all_singletons = np.arange(len(X))
    assert sampled_silhouette(X, all_singletons, seed=0) == 0.0

    # Two genuine blobs must beat the degenerate all-singleton labelling.
    centers = rng.normal(size=(2, 8))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    Y = np.repeat(centers, 6, axis=0) + 0.02 * rng.normal(size=(12, 8))
    Y = (Y / np.linalg.norm(Y, axis=1, keepdims=True)).astype(np.float32)
    two_blobs = np.repeat([0, 1], 6)
    assert sampled_silhouette(Y, two_blobs, seed=0) > sampled_silhouette(Y, all_singletons, seed=0)
