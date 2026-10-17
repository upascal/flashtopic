"""Codebook drafter: evidence, id resolution, split ids, rendering, freezing (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from flashtopic.prevalence.codebook import Codebook, build_codebook, check_frozen, freeze, render
from flashtopic.prevalence.draft import (
    CodebookDraft, DraftedCodebook, DraftEvidence, _user_prompt, build_evidence, nearest_topics,
    to_result, topic_for_codebook,
)
from flashtopic.prevalence.topics import Topic


def _topic(tid, members, label="a label", typ=("core one", "core two"), cov=("edge",)):
    return Topic(tid, list(members), label=label, typical_exemplars=list(typ), exemplars=list(cov),
                 coherence={"median": 0.5, "p10": 0.3}, label_coherence="mixed",
                 discerned_subtopics=["x", "y"])


def _dc(label, pos, neg):
    return DraftedCodebook(label=label, definition=f"About {label}.", include=["does X"],
                           exclude=["passing mention"], positive_ids=pos, negative_ids=neg)


def test_nearest_topics_uses_centroids():
    E = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1], [0.7, 0.7]], dtype=np.float32)
    ts = [Topic("A", [0, 1]), Topic("B", [2, 3]), Topic("C", [4])]
    near = nearest_topics(ts, E, n=1)
    assert near["A"][0].topic_id == "C" and near["B"][0].topic_id == "C"


def test_evidence_orders_core_first_and_prompt_numbers_everything():
    E = np.eye(3, dtype=np.float32)
    ts = [_topic("T1", [0]), _topic("T2", [1], label="neighbour", typ=("nb sentence",))]
    ev = build_evidence(ts, E, n_docs=10, member_doc_counts={"T1": 3})["T1"]
    assert ev.members == ["core one", "core two", "edge"] and ev.naive_rate == 0.3
    p = _user_prompt(ev)
    assert "S1 (core) core one" in p and "S3 edge" in p and "N1 nb sentence" in p
    assert "Labeler's verdict: mixed — facets: x; y" in p


def test_split_gets_suffixed_ids_and_examples_resolve_to_real_sentences():
    ev = DraftEvidence(_topic("T7", [0]), ["s1", "s2", "s3"], [("nb", ["n1", "n2"])])
    d = CodebookDraft(decision="split", rationale="two ideas", label_fits_cluster=False,
                      breadth_warning=None,
                      codebooks=[_dc("first", ["S1", "S9", "N1"], ["N2", "S3"]), _dc("second", ["s2"], [])])
    r = to_result(ev, d)
    a, b = r.codebooks
    assert (a.topic_id, b.topic_id) == ("T7a", "T7b") and a.source_topic_id == "T7"
    assert a.positive_examples == ["s1"]  # S9 doesn't exist; N# not allowed as a positive
    assert a.negative_examples == ["n2", "s3"] and b.positive_examples == ["s2"]
    assert "did not fit" in a.notes
    assert topic_for_codebook(ev.topic, b).topic_id == "T7b"


def test_reject_and_error_yield_no_codebooks():
    ev = DraftEvidence(_topic("T1", [0]), ["s1"])
    d = CodebookDraft(decision="reject", rationale="noise", label_fits_cluster=True,
                      breadth_warning=None, codebooks=[_dc("x", ["S1"], [])])
    assert to_result(ev, d).codebooks == []
    assert to_result(ev, None, error="boom").decision == "error"


def test_template_codebooks_render_unchanged_and_drafted_ones_show_rules():
    t = Topic("T1", [0], label="topic", typical_exemplars=["example"])
    plain = build_codebook(t)
    assert "COUNTS" not in render(plain) and "NEGATIVE" not in render(plain)
    rich = Codebook("T1", "topic", "def", ["p"], include=["i"], exclude=["e"], negative_examples=["n"])
    r = render(rich)
    assert "COUNTS when the document:\n  - i" in r and "DOES NOT COUNT:\n  - e" in r
    assert "NEGATIVE examples" in r and "  - n" in r


def test_freeze_detects_edits():
    cb = freeze(Codebook("T1", "topic", "def", ["p"]))
    check_frozen(cb)
    cb.definition = "edited after seeing gold"
    with pytest.raises(ValueError):
        check_frozen(cb)
