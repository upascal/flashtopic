"""Orchestration test for label_hierarchy.label_levels (no real LLM).

Stubs the structured LLM call to echo child refs back, then asserts the driver groups
children under parents, batches oversized groups, threads refs through, and returns a
label for every cluster at every tier.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np
import pytest


def _stub_call(monkeypatch):
    from flashtopic import label as label_mod
    from flashtopic.label import ContrastiveChild, ContrastiveGroup
    from flashtopic.llm import LLMResult

    async def fake_structured(system, user, model, schema, **kw):
        refs = []
        for r in re.findall(r"\[(\d+:\d+)\]", user):  # only child headers look like [ci:cid]
            if r not in refs:
                refs.append(r)
        grp = ContrastiveGroup(
            parent_theme="theme",
            parent_unifies="unifies",
            children=[ContrastiveChild(ref=r, label=f"label-{r}", differs="d") for r in refs],
        )
        return grp, LLMResult("", model, 10, 10, 0.0, None)

    monkeypatch.setattr(label_mod, "call_llm_structured", fake_structured)


def test_label_levels_labels_every_tier(monkeypatch):
    _stub_call(monkeypatch)
    from flashtopic.label_hierarchy import label_levels

    # 8 uniques, one per fine cluster. 3 tiers: 2 super, 4 meta, 8 fine.
    fine = np.arange(8)
    meta = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    sup = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    parents = [{0: 0, 1: 0, 2: 1, 3: 1}, {i: i // 2 for i in range(8)}]  # meta→super, fine→meta
    E = np.random.default_rng(0).normal(size=(8, 16)).astype(np.float32)

    terms_by_tier = {ci: {c: [f"t{ci}-{c}"] for c in range(n)} for ci, n in {0: 2, 1: 4, 2: 8}.items()}
    exemplars_by_tier = {ci: {c: [f"ex{ci}-{c}"] for c in d} for ci, d in terms_by_tier.items()}

    out, entries = asyncio.run(
        label_levels(
            E=E, level_labels=[sup, meta, fine], parents=parents,
            terms_by_tier=terms_by_tier, exemplars_by_tier=exemplars_by_tier,
            corpus_topic="x", model="fake/model",
            level_names=["category", "topic", "subtopic"], batch_size=1,  # force batching
        )
    )

    # Every cluster at every tier got a label echoing its ref.
    assert set(out[2]) == set(range(8)) and out[2][5]["label"] == "label-2:5"
    assert set(out[1]) == set(range(4))
    assert set(out[0]) == {0, 1}
    # Parent ids wired correctly.
    assert out[2][5]["parent_id"] == 2 and out[1][2]["parent_id"] == 1
    assert out[0][0]["parent_id"] is None  # coarsest tier → synthetic root
    assert out[2][3]["parent_theme"] == "theme"
    assert entries  # cache entries produced for persistence


def test_label_levels_flat_single_root(monkeypatch):
    _stub_call(monkeypatch)
    from flashtopic.label_hierarchy import label_levels

    fine = np.array([0, 1, 2, 3, 4])
    E = np.random.default_rng(1).normal(size=(5, 8)).astype(np.float32)
    terms = {0: {c: [f"t{c}"] for c in range(5)}}
    ex = {0: {c: [f"e{c}"] for c in range(5)}}

    out, entries = asyncio.run(
        label_levels(
            E=E, level_labels=[fine], parents=[],
            terms_by_tier=terms, exemplars_by_tier=ex,
            corpus_topic="x", model="fake/model", level_names=["subtopic"], batch_size=8,
        )
    )
    assert set(out[0]) == set(range(5))
    assert all(v["parent_id"] is None for v in out[0].values())  # all under root
