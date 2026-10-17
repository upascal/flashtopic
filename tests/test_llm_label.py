"""LLM gateway (secrets, parsing) + labeling (prompts, batching) tests.

All offline except an optional live atom-label call gated on OPENROUTER_API_KEY.
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest

import flashtopic.llm as llm
from flashtopic.label import (
    AtomEvidence,
    _atom_system_prompt,
    _group_system_prompt,
    batch_by_similarity,
)


# ---- gateway / secrets ----------------------------------------------------- #
def test_configure_llm_overrides_and_builds_client():
    llm.configure_llm(api_key="sk-test-key", base_url="https://example.test/v1")
    client = llm.get_client()
    assert "example.test" in str(client.base_url)
    llm.configure_llm(api_key=os.environ.get("OPENROUTER_API_KEY"))  # restore


def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    llm._config["api_key"] = None
    llm._client = None
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llm.get_client()


def test_cost_and_json_parsing():
    assert llm.estimate_cost_usd("anthropic/claude-haiku-4.5", 1_000_000, 1_000_000) == 6.0
    assert llm.estimate_cost_usd("unknown/model", 1_000_000, 0) == 0.0
    assert llm.parse_json_from_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json_from_text('{"b": 2}') == {"b": 2}


# ---- labeling: prompts + batching (offline) -------------------------------- #
def test_atom_prompt_injects_corpus_topic():
    p = _atom_system_prompt("the city transit strike")
    assert "the city transit strike" in p and "do NOT label at" in p


def test_atom_prompt_degrades_cleanly_without_topic():
    # The general-corpus default: no driving question. The prompt must read as plain
    # cluster labeling — never a dangling "about ." and no leftover "umbrella" jargon.
    p = _atom_system_prompt("")
    assert "about ." not in p
    assert "umbrella" not in p.lower()
    assert "text corpus" in p and "SPECIFIC topic" in p


def test_group_prompt_uses_levels_and_topic():
    p = _group_system_prompt("topic", "corpus", "wildfire policy")
    assert "topics" in p and "corpus" in p and "wildfire policy" in p


def test_group_prompt_degrades_cleanly_without_topic():
    p = _group_system_prompt("topic", "corpus", "")
    assert "about ." not in p and "umbrella" not in p.lower()


def test_batch_by_similarity_partitions_all_ids():
    ids = [f"c{i}" for i in range(10)]
    med = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    med /= np.linalg.norm(med, axis=1, keepdims=True)
    batches = batch_by_similarity(ids, med, n=4)
    assert all(len(b) <= 4 for b in batches)
    assert sorted(x for b in batches for x in b) == sorted(ids)


def test_batch_by_similarity_single_batch_when_small():
    ids = ["a", "b", "c"]
    med = np.eye(3, dtype=np.float32)
    assert batch_by_similarity(ids, med, n=8) == [["a", "b", "c"]]


# ---- optional live atom label --------------------------------------------- #
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
def test_live_atom_label():
    from flashtopic.label import label_atom

    ev = AtomEvidence(
        atom_id=0,
        terms=["strike", "union", "wages"],
        exemplars=["Transit workers voted to strike over stalled wage talks.",
                   "Calls for a walkout grew as negotiations dragged on."],
    )
    parsed, raw = asyncio.run(
        label_atom(ev, model="anthropic/claude-haiku-4.5", corpus_topic="a city transit dispute")
    )
    if raw.error and ("Connection" in raw.error or "llm_call_failed" in raw.error):
        pytest.skip(f"API unreachable in this environment: {raw.error}")
    assert raw.error is None and parsed is not None
    assert parsed.specific_topic and parsed.coherence in {"coherent", "mixed", "noise"}
