"""``_cluster_evidence`` samples each cluster's siblings independently of the others.

Pins the fix for a shared-generator defect: one ``default_rng(seed)`` above the loop
meant a cluster's sibling sample depended on how many random draws the clusters
scored before it had consumed, so its distinctive terms moved with the *sizes and
order* of unrelated clusters. Needs no spaCy, model or backend: ``distinctive_terms``
is stubbed to record what it was handed.
"""

from __future__ import annotations

import numpy as np
import pytest

import flashtopic.pipeline as pipeline
from flashtopic.corpus import UniqueIndex
from flashtopic.pipeline import _SIBLING_CAP, _cluster_evidence


def _index(n: int) -> UniqueIndex:
    return UniqueIndex(
        texts=[f"sentence {i}" for i in range(n)],
        sentence_idxs=[[i] for i in range(n)],
        doc_idxs=[[i % 7] for i in range(n)],
    )


def _sibling_texts_seen(monkeypatch, labels: np.ndarray, *, seed: int) -> dict[int, list[str]]:
    seen: dict[int, list[str]] = {}
    calls = iter(sorted(set(labels.tolist())))

    def fake_terms(member_texts, sib_texts, **_):
        seen[next(calls)] = list(sib_texts)
        return []

    monkeypatch.setattr(pipeline, "distinctive_terms", fake_terms)
    n = len(labels)
    E = np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32)
    _cluster_evidence(
        labels, _index(n), E,
        domain_stopwords=(), n_exemplars=2, n_terms=5, seed=seed,
    )
    return seen


def _labels(size_0: int, size_1: int, size_2: int) -> np.ndarray:
    return np.array([0] * size_0 + [1] * size_1 + [2] * size_2)


def test_a_clusters_sibling_sample_ignores_the_draws_made_before_it(monkeypatch):
    # Cluster 2 is identical in both layouts (same members, same non-members); only
    # the split between clusters 0 and 1 moves. In the first layout both 0 and 1
    # exceed the cap and draw; in the second, cluster 0's non-members fit under the
    # cap and it draws nothing. With one shared generator that changed what cluster
    # 2 was handed; with one generator per cluster it must not.
    assert _SIBLING_CAP == 1500
    first = _labels(1500, 1600, 100)
    second = _labels(1700, 1400, 100)
    for layout in (first, second):
        assert (layout == 2).sum() == 100 and len(layout) == 3200
    a = _sibling_texts_seen(monkeypatch, first, seed=11)
    b = _sibling_texts_seen(monkeypatch, second, seed=11)
    assert len(a[2]) == len(b[2]) == _SIBLING_CAP
    assert a[2] == b[2]


def test_the_sample_is_still_seeded(monkeypatch):
    labels = _labels(1500, 1600, 100)
    once = _sibling_texts_seen(monkeypatch, labels, seed=3)
    again = _sibling_texts_seen(monkeypatch, labels, seed=3)
    other = _sibling_texts_seen(monkeypatch, labels, seed=4)
    assert once == again
    assert once[2] != other[2]


def test_small_clusters_are_handed_every_sibling(monkeypatch):
    seen = _sibling_texts_seen(monkeypatch, _labels(20, 30, 10), seed=0)
    assert [len(seen[c]) for c in (0, 1, 2)] == [40, 30, 50]


@pytest.mark.parametrize("seed", [None])
def test_an_unseeded_run_still_runs(monkeypatch, seed):
    seen = _sibling_texts_seen(monkeypatch, _labels(5, 5, 5), seed=seed)
    assert set(seen) == {0, 1, 2}
