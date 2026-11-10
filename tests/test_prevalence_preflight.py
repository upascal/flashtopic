"""Codebook test items run through the classifier, and the CV cut-off."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from flashtopic.prevalence.classifier import Answer, MockClassifier
from flashtopic.prevalence.codebook import Codebook
from flashtopic.prevalence.preflight import calibrate_cutoff, check_items, preflight
from flashtopic.prevalence.questions import build_question_set
from flashtopic.prevalence.topics import Topic


def _cb():
    return Codebook(topic_id="T1", label="strike votes", definition="d",
                    positive_examples=["Members voted to strike.", "The vote passed."],
                    negative_examples=["Fares rose.", "Members voted to strike."])


def test_check_items_keep_expectations_and_dedup():
    items = check_items(_cb())
    assert [(i.text, i.expected) for i in items] == [
        ("Members voted to strike.", True), ("The vote passed.", True), ("Fares rose.", False)]
    assert all(i.doc_id < 0 for i in items) and len({i.doc_id for i in items}) == 3


def test_preflight_scores_items_and_reports_near_misses(tmp_path):
    def rule(text, q):
        return Answer(q.qid, p_yes=0.9 if "vote" in text.lower() else 0.1)

    t = Topic(topic_id="T1", member_uids=[0], label="strike votes")
    qs = build_question_set([t], codebooks={"T1": _cb()})
    res = asyncio.run(preflight(_cb(), qs, MockClassifier(rule), cache_dir=tmp_path))
    assert res.n_pass == 3 and res.hard_negative_p == [0.1] and not res.failures()
    assert "3/3" in res.summary()


def test_calibrate_cutoff_finds_a_high_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    # Positives sit above 0.75, negatives spread up to 0.7: the default 0.5 over-counts.
    p = np.where(y == 1, rng.uniform(0.76, 1.0, 400), rng.uniform(0.0, 0.7, 400))
    cut, acc = calibrate_cutoff(p, y)
    assert 0.69 <= cut <= 0.77 and acc > 0.97
    with pytest.raises(ValueError):
        calibrate_cutoff([0.5, np.nan], [1, 0])
