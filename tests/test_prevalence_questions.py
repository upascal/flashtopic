"""Question generation is deterministic, pre-registered (topic structure only), hash-stable."""

from __future__ import annotations

import pytest

from flashtopic.prevalence.codebook import build_codebook
from flashtopic.prevalence.questions import (
    DEPTH_LEVELS, NONE_OPTION, TopicContext, build_question_set, depth_question,
    facet_questions, parent_question, presence_question, question_set_hash, role_of, sibling_question,
)
from flashtopic.prevalence.topics import Topic


def _topic(i, label, group=None):
    return Topic(topic_id=f"T{i}", member_uids=[i], label=label,
                 distinctive_terms=[f"{label}ish", "thing"], exemplars=[f"about {label}"])


def test_presence_and_depth_carry_the_codebook_as_criteria():
    t = _topic(1, "strike votes")
    cb = build_codebook(t, corpus_topic="a labor dispute")
    p = presence_question(t, cb)
    d = depth_question(t, cb)
    assert p.kind == "noul" and p.qid == "T1.presence"
    assert "strike votes" in p.text and cb.definition in p.criteria
    assert d.kind == "score" and d.options == DEPTH_LEVELS and d.criteria == p.criteria
    assert role_of(p.qid) == "presence" and role_of(d.qid) == "depth"


def test_noul_questions_keep_passing_mention_rule_out_of_the_wording():
    # jev-1.13 A/B: "substantively ... beyond a passing mention?" crushed true positives;
    # the rule belongs in criteria_false, the question stays plain.
    t = _topic(1, "strike votes")
    qs = [presence_question(t), parent_question("g1", "the transit strike", ("strike votes", "fare changes")),
          *facet_questions(t, ("labor unions",))]
    for q in qs:
        assert q.kind == "noul" and q.criteria and "passing" in q.criteria_false
        assert "substantively" not in q.text and "passing" not in q.text
    parent = qs[1]
    assert parent.text == "Does the document discuss the transit strike?"
    assert "strike votes" in parent.criteria and "fare changes" in parent.criteria


def test_build_question_set_gives_parent_the_sibling_facets():
    ctx = TopicContext(meta_theme="the transit strike", group_id="g1",
                       sibling_labels=("strike votes", "fare changes"))
    qs = build_question_set([_topic(1, "strike votes")], {"T1": ctx})
    assert "fare changes" in qs.question("group:g1.parent").criteria


def test_unlabeled_topic_is_rejected():
    with pytest.raises(ValueError):
        presence_question(Topic(topic_id="T9", member_uids=[0]))


def test_sibling_question_has_none_option_and_caps_at_255():
    q = sibling_question("g1", ("a", "b", "a"), meta_theme="stuff")
    assert q.kind == "choice" and q.options == ("a", "b", NONE_OPTION)
    with pytest.raises(ValueError):
        sibling_question("g2", tuple(str(i) for i in range(300)))


def test_build_question_set_dedups_group_questions_and_maps_topics():
    ts = [_topic(1, "bus routes"), _topic(2, "night trains"), _topic(3, "strike")]
    ctx = {
        "T1": TopicContext(meta_theme="fare changes", group_id="g", sibling_labels=("bus routes", "night trains")),
        "T2": TopicContext(meta_theme="fare changes", group_id="g", sibling_labels=("bus routes", "night trains")),
        "T3": TopicContext(facets=("threats", "withdrawals")),
    }
    qs = build_question_set(ts, ctx)
    qids = [q.qid for q in qs.questions]
    assert qids.count("group:g.parent") == 1 and qids.count("group:g.sibling") == 1
    assert qs.by_topic["T1"] == ["T1.presence", "T1.depth", "group:g.parent", "group:g.sibling"]
    assert qs.by_topic["T2"][2:] == qs.by_topic["T1"][2:]  # shared
    assert qs.by_topic["T3"] == ["T3.presence", "T3.depth", "T3.facet1", "T3.facet2"]
    assert qs.choice_option == {"T1": "bus routes", "T2": "night trains"}
    assert len(qs.for_topic("T1")) == 4


def test_hash_is_order_independent_and_content_sensitive():
    ts = [_topic(1, "a"), _topic(2, "b")]
    h1 = build_question_set(ts).hash
    h2 = build_question_set(ts[::-1]).hash
    assert h1 == h2 == question_set_hash(build_question_set(ts).questions)
    ts[0].label = "a (revised)"
    assert build_question_set(ts).hash != h1
