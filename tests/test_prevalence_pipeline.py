"""End-to-end prevalence wiring: clustering → topics → surrogate → design → estimate.

No LLM (the judge is the same gateway already covered elsewhere); gold labels are
synthesized to exercise the data → surrogate → sampling → estimator path.
"""

from __future__ import annotations

import numpy as np

from flashtopic.prevalence import (
    RunContext,
    build_surrogate,
    design_sample,
    estimate_prevalence,
    topics_from_labels,
)


def _ctx(n_uniq=300, n_docs=150, d=16, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n_uniq, d)).astype(np.float32)
    uid_docs = [sorted(rng.choice(n_docs, int(rng.integers(1, 4)), replace=False).tolist())
                for _ in range(n_uniq)]
    texts = [f"unique sentence {i}" for i in range(n_uniq)]
    doc_texts = {i: f"document {i} body text" for i in range(n_docs)}
    return RunContext(E=E, texts=texts, uid_docs=uid_docs, n_docs=n_docs, doc_texts=doc_texts), rng


def test_runcontext_inverts_membership():
    ctx, _ = _ctx()
    # every document that a uid lists must list that uid back in doc_uids
    for uid, docs in enumerate(ctx.uid_docs):
        for di in docs:
            assert uid in ctx.doc_uids[di]
    assert np.allclose(np.linalg.norm(ctx.E, axis=1), 1.0, atol=1e-4)  # normalized on load


def test_runcontext_from_result_reconstructs_doc_texts():
    from flashtopic.corpus import SentenceRecord, UniqueIndex
    from flashtopic.pipeline import ClusterResult

    records = [
        SentenceRecord(0, 0, "d0", (), "body", "alpha sentence", 2),
        SentenceRecord(1, 0, "d0", (), "body", "beta sentence", 2),
        SentenceRecord(2, 1, "d1", (), "body", "alpha sentence", 2),
    ]
    ui = UniqueIndex(texts=["alpha sentence", "beta sentence"],
                     sentence_idxs=[[0, 2], [1]], doc_idxs=[[0, 1], [0]])
    res = ClusterResult(labels=np.array([0, 1]), topics=[], unique_index=ui,
                        embeddings=np.eye(2, dtype=np.float32), records=records, k=2, n_docs=2)

    ctx = RunContext.from_result(res)
    assert ctx.n_docs == 2
    assert ctx.doc_texts[0] == "alpha sentence\nbeta sentence"  # sentence-order join
    assert ctx.doc_texts[1] == "alpha sentence"
    assert ctx.uid_docs == [[0, 1], [0]]


def test_topics_from_labels_partition_uids():
    labels = np.array([0, 1, 0, 2, 1, 2, 0])
    topics = topics_from_labels(labels)
    assert {t.topic_id for t in topics} == {"T0", "T1", "T2"}
    all_uids = sorted(u for t in topics for u in t.member_uids)
    assert all_uids == list(range(7))


def test_surrogate_and_estimate_with_perfect_gold():
    ctx, rng = _ctx()
    labels = rng.integers(0, 6, size=len(ctx.texts))
    topic = max(topics_from_labels(labels), key=lambda t: t.n_members)

    sur = build_surrogate(ctx, topic)
    assert sur.f_memb.shape == (ctx.n_docs,)
    assert 0.0 <= sur.naive <= 1.0
    assert sur.score.shape == (ctx.n_docs,)

    design = design_sample(sur)
    assert design.n_gold > 0
    # perfect gold (= surrogate) → debiased prevalence ≈ naive
    y = design.f_s.copy()
    res = estimate_prevalence(topic.topic_id, sur.f_memb.astype(int),
                              design.f_s, y, design.pi_s, design.strata, seed=0)
    assert abs(res.pi_ppi - sur.naive) < 0.05
    assert res.n_docs == ctx.n_docs
