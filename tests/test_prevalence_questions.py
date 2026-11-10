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


def _drafted(t):
    from flashtopic.prevalence.codebook import Codebook
    return Codebook(
        topic_id=t.topic_id, label=t.label, definition="Argues strike votes decide the outcome.",
        positive_examples=["The vote settled it."],
        include=["Says the vote decided the strike", "Explains why members voted"],
        exclude=["Fare changes, unless tied to the vote",
                 "The topic appears only as a bare item in a list, or as a single passing phrase."],
        negative_examples=["Fares rose in March."],
    )


def test_drafted_rules_and_near_misses_reach_the_presence_question():
    t = _topic(1, "strike votes")
    cb = _drafted(t)
    p = presence_question(t, cb)
    assert "Says the vote decided the strike" in p.criteria
    assert "Fare changes, unless tied to the vote" in p.criteria_false
    assert "Fares rose in March." in p.criteria_false and "passing" in p.criteria_false
    assert p.text == "Does the document discuss strike votes?"  # criteria placement: plain wording


def test_template_codebook_questions_are_unchanged_by_the_drafted_sections():
    # No include/exclude/negatives → the same text as before, so cached answers still hit.
    t = _topic(1, "strike votes")
    p = presence_question(t, build_codebook(t))
    assert "It counts when" not in p.criteria and p.criteria_false.count("\n") == 0


def test_codebook_placement_moves_the_codebook_into_the_instructions():
    t = _topic(1, "strike votes")
    cb = _drafted(t)
    ins = presence_question(t, cb, codebook_in="instructions")
    both = presence_question(t, cb, codebook_in="both")
    assert "Fare changes" in ins.text and "Fare changes" not in ins.criteria_false
    assert "Fare changes" in both.text and "Fare changes" in both.criteria_false
    assert len({presence_question(t, cb).content_hash, ins.content_hash, both.content_hash}) == 3
    assert "Fare changes" in depth_question(t, cb, codebook_in="instructions").criteria
    with pytest.raises(ValueError):
        presence_question(t, cb, codebook_in="everywhere")


def test_rule_and_nearmiss_checks_come_from_the_codebook_without_the_passing_rule():
    t = _topic(1, "strike votes")
    qs = build_question_set([t], codebooks={"T1": _drafted(t)})
    assert qs.by_topic["T1"] == ["T1.presence", "T1.depth", "T1.rule1", "T1.rule2", "T1.nearmiss1"]
    assert [role_of(q) for q in qs.by_topic["T1"][2:]] == ["rule", "rule", "nearmiss"]
    assert "Fare changes" in qs.question("T1.nearmiss1").text
    assert build_question_set([t], codebooks={"T1": _drafted(t)}, include_rules=False).by_topic["T1"] == [
        "T1.presence", "T1.depth"]
