"""Nested multi-resolution clustering: flat measures nesting, dendrogram guarantees it.

The nesting-math unit test is backend-free; the cluster_levels tests need a clustering
backend (skipped otherwise).
"""

from __future__ import annotations

import numpy as np
import pytest

from flashtopic.cluster import available_backends


def _blobs(k=12, n_per=40, d=16, seed=0):
    """k well-separated blobs on the unit sphere → n = k*n_per points."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = np.repeat(centers, n_per, axis=0) + 0.05 * rng.normal(size=(k * n_per, d))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def test_nesting_metrics_perfect_vs_tangled():
    """Backend-free: the ported nesting math behaves as documented."""
    from flashtopic.cluster.levels import nested_contingency, nesting_metrics

    coarse = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    nested = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # each fine sits in one coarse
    nm = nesting_metrics(coarse, nested, coarse_k=2, fine_k=4)
    assert nm.purity >= 0.999 and nm.homogeneity >= 0.999
    assert nested_contingency(coarse, nested).shape == (2, 4)

    tangled = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # each fine spans both coarse
    nm2 = nesting_metrics(coarse, tangled, coarse_k=2, fine_k=2)
    assert nm2.purity < 0.9


@pytest.mark.skipif(not available_backends(), reason="no clustering backend")
def test_flat_levels_measure_nesting():
    from flashtopic.cluster import cluster_levels

    E = _blobs(k=12, n_per=40)
    res = cluster_levels(E, strategy="flat", levels=[3, 6, 12], seed=0)

    assert res.strategy == "flat"
    assert len(res.labels) == 3 and len(res.parents) == 2 and len(res.nesting) == 2
    assert res.ks == sorted(res.ks)  # coarse → fine
    for lab in res.labels:
        assert lab.shape == (E.shape[0],)
    # parents[i] is indexed by fine cluster id, one entry per fine cluster
    assert res.parents[0].shape[0] == res.ks[1]
    for nm in res.nesting:
        assert 0.0 <= nm.purity <= 1.0 and 0.0 <= nm.homogeneity <= 1.0


@pytest.mark.skipif(not available_backends(), reason="no clustering backend")
def test_dendrogram_levels_nest_perfectly():
    from flashtopic.cluster import cluster_levels

    E = _blobs(k=12, n_per=40)
    res = cluster_levels(E, strategy="dendrogram", cuts=[4, 12, 40], seed=0)

    assert res.strategy == "dendrogram" and res.linkage is not None
    assert len(res.labels) == 3 and res.ks == sorted(res.ks)
    # nested by construction → every fine cluster has exactly one coarse parent
    for nm in res.nesting:
        assert nm.purity >= 0.999
    # the parent map agrees with the actual coarse labels for every point
    for i in range(len(res.labels) - 1):
        coarse, fine = res.labels[i], res.labels[i + 1]
        assert np.all(res.parents[i][fine] == coarse)


@pytest.mark.skipif(not available_backends(), reason="no clustering backend")
def test_cluster_levels_validates_args():
    from flashtopic.cluster import cluster_levels

    E = _blobs(k=4, n_per=20)
    with pytest.raises(ValueError, match="needs levels"):
        cluster_levels(E, strategy="flat")
    with pytest.raises(ValueError, match="atom_k"):
        cluster_levels(E, strategy="dendrogram", cuts=[5], atom_k=3)


def test_prune_small_clusters_absorbs_singletons():
    """Backend-free: tiny clusters roll into the nearest surviving centroid."""
    from flashtopic.cluster import prune_small_clusters

    X = np.vstack([np.tile([1.0, 0.0], (3, 1)),
                   np.tile([0.0, 1.0], (3, 1))]).astype(np.float32)
    labels = np.array([0, 0, 0, 1, 1, 2])  # cluster 2 is a singleton (a "B" point)
    out = prune_small_clusters(labels, X, min_size=2)

    sizes = np.bincount(out)
    assert len(sizes) == 2 and sizes.min() >= 2     # singleton gone
    assert out[5] == out[3]                          # absorbed into the other B cluster
    assert out[0] != out[3]                          # A and B stay distinct


@pytest.mark.skipif(not available_backends(), reason="no clustering backend")
def test_n_init_keeps_lowest_inertia():
    from flashtopic.cluster import spherical_kmeans

    E = _blobs(k=12, n_per=40)
    r1 = spherical_kmeans(E, 20, seed=0, n_init=1)
    r5 = spherical_kmeans(E, 20, seed=0, n_init=5)
    assert r5.inertia <= r1.inertia + 1e-6  # restarts can only help


@pytest.mark.skipif(not available_backends(), reason="no clustering backend")
def test_dendrogram_min_size_prunes_and_preserves_nesting():
    from flashtopic.cluster import cluster_levels

    E = _blobs(k=12, n_per=40)
    res = cluster_levels(E, strategy="dendrogram", cuts=[4, 12, 40],
                         atom_k=60, min_size=5, seed=0)
    for lab in res.labels:
        assert np.bincount(lab).min() >= 5  # no trivia at any level
    for nm in res.nesting:
        assert nm.purity >= 0.999  # atom-base pruning preserves perfect nesting


@pytest.mark.skipif(not available_backends(), reason="no clustering backend")
def test_level_stability_reports_per_level_ari():
    from flashtopic.cluster import level_stability

    E = _blobs(k=8, n_per=50)
    rep = level_stability(E, strategy="dendrogram", cuts=[8, 24], atom_k=40,
                          n_init=3, seeds=(0, 1, 2))
    assert len(rep.mean_ari) == 2 and len(rep.min_ari) == 2 and len(rep.ks) == 2
    assert all(-1.0 <= a <= 1.0 for a in rep.mean_ari)
    assert rep.mean_ari[0] >= 0.5  # coarse level over 8 clean blobs is reproducible


def _hier_blobs(n_super=4, n_meta=3, n_leaf=3, n_per=40, d=32, seed=0):
    """A known 3-level hierarchy: n_super → n_super*n_meta → n_super*n_meta*n_leaf."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_super):
        sc = rng.normal(size=d)
        sc /= np.linalg.norm(sc)
        for _ in range(n_meta):
            mc = sc + 0.20 * rng.normal(size=d)
            mc /= np.linalg.norm(mc)
            for _ in range(n_leaf):
                lc = mc + 0.18 * rng.normal(size=d)
                lc /= np.linalg.norm(lc)
                p = lc + 0.06 * rng.normal(size=(n_per, d))
                pts.append(p / np.linalg.norm(p, axis=1, keepdims=True))
    return np.vstack(pts).astype(np.float32)


@pytest.mark.skipif(not available_backends(), reason="no clustering backend")
def test_select_ladder_recovers_a_sane_hierarchy():
    from flashtopic.cluster import select_ladder
    from flashtopic.cluster.levels import DEFAULT_LEVELS_CONFIG as CFG

    E = _hier_blobs(n_super=4, n_meta=3, n_leaf=3)  # truth ≈ 4 → 12 → 36
    sel = select_ladder(E, n_levels=3, seed=0)

    assert len(sel.ks) == 3
    assert sel.ks == sorted(sel.ks) and len(set(sel.ks)) == 3  # strictly increasing
    for coarse, fine in zip(sel.ks, sel.ks[1:]):
        assert fine / coarse >= CFG.min_pair_ratio  # tiers stay distinct
    assert len(sel.rows) == 3 and sel.n_atoms >= sel.ks[-1]
    # Ladder is data-driven, not pinned to one resolution: the coarse tier is a
    # handful of themes and the fine tier is an order of magnitude larger.
    assert 2 <= sel.ks[0] <= 8
    assert 20 <= sel.ks[-1] <= 60


@pytest.mark.skipif(not available_backends(), reason="no clustering backend")
def test_select_ladder_degrades_on_a_corpus_too_small_to_tier():
    """Too few atoms for 3 ratio-respecting tiers → fewer levels, not a crash."""
    from flashtopic.cluster import select_ladder

    E = _blobs(k=3, n_per=6, d=8)  # 18 points
    sel = select_ladder(E, n_levels=3, atom_k=4, seed=0)
    assert 1 <= len(sel.ks) <= 3
    assert sel.ks == sorted(sel.ks)


@pytest.mark.skipif(not available_backends(), reason="no clustering backend")
def test_select_ladder_feeds_cluster_levels():
    """The whole point: the chosen ks materialize as a real nested ladder."""
    from flashtopic.cluster import cluster_levels, select_ladder

    E = _hier_blobs(n_super=3, n_meta=3, n_leaf=3, n_per=30)
    sel = select_ladder(E, n_levels=3, seed=0)
    res = cluster_levels(E, strategy="dendrogram", cuts=sel.ks,
                         atom_k=sel.n_atoms, seed=0)
    assert res.ks == sorted(res.ks)
    for nm in res.nesting:
        assert nm.purity >= 0.999  # dendrogram cuts nest by construction
