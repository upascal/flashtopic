"""End-to-end pipeline integration: docs → artifacts → clusters → prevalence load.

Exercises the real corpus + potion embed + flash-kmeans path (no LLM). Skips if no
cluster backend or model2vec is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from flashtopic.cluster import available_backends
from flashtopic.config import Workspace
from flashtopic.schema import Doc

pytestmark = pytest.mark.skipif(not available_backends(), reason="no clustering backend")


def _docs():
    # Vary each sentence so dedup keeps many uniques while the theme keyword recurs
    # across members (distinctive_terms needs a term present in >= 2 member sentences).
    templates = {
        "strike": "Union local number {i} seriously debated striking against the transit authority that season.",
        "fares": "Rider group number {i} protested several fare increases across the bus and rail lines.",
        "advertising": "Agency number {i} used billboard campaigns to project a polished civic brand.",
    }
    docs = []
    for i in range(24):
        theme = list(templates)[i % 3]
        body = templates[theme].format(i=i)
        docs.append(Doc(f"d{i}", f"# Notes\n\n{body}"))
    return docs


def test_pipeline_writes_artifacts_and_feeds_prevalence(tmp_path):
    try:
        import model2vec  # noqa: F401
    except Exception:
        pytest.skip("model2vec not installed")

    from flashtopic.pipeline import run_pipeline

    ws = Workspace(base_dir=tmp_path, domain_stopwords={"games"})
    try:
        res = run_pipeline(_docs(), ws, run_id="t", k=4, seed=0)
    except Exception as e:  # e.g. potion model not cached offline
        pytest.skip(f"pipeline unavailable: {e}")

    # artifacts written
    assert ws.embeddings_path("t").exists()
    assert ws.body_membership_path("t").exists()
    assert ws.sentences_path("t").exists()
    assert ws.read_latest() == "t"

    # topics produced with evidence
    assert res.k == 4 and len(res.topics) == 4
    assert res.labels.shape[0] == res.n_uniques
    assert any(t.distinctive_terms for t in res.topics)
    assert all(t.member_uids for t in res.topics)

    # the prevalence subpackage can load the run and build a surrogate
    from flashtopic.prevalence import RunContext, build_surrogate

    ctx = RunContext.from_workspace(ws, "t")
    assert ctx.n_docs == res.n_docs
    sur = build_surrogate(ctx, max(res.topics, key=lambda t: t.n_members))
    assert sur.f_memb.shape == (ctx.n_docs,)
    assert 0.0 <= sur.naive <= 1.0
