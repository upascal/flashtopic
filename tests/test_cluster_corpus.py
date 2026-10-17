"""Storeless core: cluster_corpus returns in-memory results and touches no disk.

Uses precomputed ``embeddings=`` + a fake clusterer so it needs neither an embedding
model nor a GPU backend — only spaCy for the corpus build (skipped if absent).
"""

from __future__ import annotations

import numpy as np
import pytest

from flashtopic.schema import Doc


def _has_spacy() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_spacy(), reason="spacy + en_core_web_sm required")


class _FakeClusterer:
    """Deterministic round-robin clusterer (no backend needed); conforms to Clusterer."""

    def fit(self, X, k, *, seed=0):
        labels = (np.arange(X.shape[0]) % k).astype(np.int64)
        return type("R", (), {"labels": labels})()


def _docs(n=12):
    return [Doc(f"d{i}", f"# Notes\n\nThe theme keyword recurs in document number {i} here today.")
            for i in range(n)]


def test_cluster_corpus_is_storeless(tmp_path, monkeypatch):
    from flashtopic.corpus import build_sentence_corpus, build_unique_index
    from flashtopic.pipeline import ClusterResult, cluster_corpus

    monkeypatch.chdir(tmp_path)  # any stray write would land (and be detected) here
    docs = _docs()
    ui = build_unique_index(build_sentence_corpus(docs), "body")
    n, d = len(ui.texts), 16
    rng = np.random.default_rng(0)
    E = rng.normal(size=(n, d)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    res = cluster_corpus(docs, embeddings=E, clusterer=_FakeClusterer(), k=3, seed=0)

    assert isinstance(res, ClusterResult)
    assert res.labels.shape == (n,)
    assert len(res.topics) == 3
    assert res.embeddings.shape == (n, d)
    assert res.records and res.n_docs == 12
    # the whole point: no Workspace, no files
    assert list(tmp_path.iterdir()) == []


def test_cluster_corpus_rejects_mismatched_embeddings():
    from flashtopic.pipeline import cluster_corpus

    with pytest.raises(ValueError, match="unique sentences"):
        cluster_corpus(_docs(4), embeddings=np.zeros((3, 8), np.float32),
                       clusterer=_FakeClusterer(), k=2)


def test_from_result_feeds_prevalence(tmp_path, monkeypatch):
    from flashtopic.corpus import build_sentence_corpus, build_unique_index
    from flashtopic.pipeline import cluster_corpus
    from flashtopic.prevalence import RunContext, build_surrogate

    monkeypatch.chdir(tmp_path)
    docs = _docs()
    ui = build_unique_index(build_sentence_corpus(docs), "body")
    rng = np.random.default_rng(1)
    E = rng.normal(size=(len(ui.texts), 16)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    res = cluster_corpus(docs, embeddings=E, clusterer=_FakeClusterer(), k=4, seed=0)
    ctx = RunContext.from_result(res)

    assert ctx.n_docs == res.n_docs
    sur = build_surrogate(ctx, max(res.topics, key=lambda t: t.n_members))
    assert sur.f_memb.shape == (ctx.n_docs,)
    assert 0.0 <= sur.naive <= 1.0
