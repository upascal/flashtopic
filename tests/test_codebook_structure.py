import pytest

from flashtopic.prevalence.codebook import (Codebook, Component, describe_formula, evaluate,
                                            formula_ids, from_markdown, parse_formula,
                                            soft_evaluate, to_markdown)
from flashtopic.prevalence.questions import build_question_set, role_of
from flashtopic.prevalence.topics import Topic


def _cb(**kw):
    return Codebook(
        topic_id="T1", label="Remote work credited with better wellbeing", definition="Proposes it.",
        positive_examples=["pos"], negative_examples=["neg"],
        components=[Component("W", "Names a wellbeing change.", yes_when=["implied"]),
                    Component("F", "Mentions remote work.", no_when=["office work in general"]),
                    Component("R", "F is presented as the cause of W.", kind="relation", links=["W", "F"]),
                    Component("B1", "Productivity statement.")],
        formula="W & F & R", near_misses={"productivity without the link": "B1 & ~R"}, **kw)


def test_formula_precedence_and_evaluation():
    assert parse_formula("A | B & ~C") == ("or", "A", ("and", "B", ("not", "C")))
    assert formula_ids("(A | B) & ~C") == {"A", "B", "C"}
    assert evaluate("A & ~B", {"A": True, "B": False})
    assert not evaluate("(A | B) & C", {"A": True, "B": False, "C": False})
    assert soft_evaluate("A & B", {"A": 0.5, "B": 0.5}) == pytest.approx(0.25)
    assert describe_formula("A & B & (C | ~D)") == "[A] AND [B] AND ([C] OR NOT [D])"


@pytest.mark.parametrize("bad", ["A &", "A B", "(A", "A + B", "~"])
def test_malformed_formula_raises(bad):
    with pytest.raises(ValueError):
        parse_formula(bad)


def test_structured_codebook_validates_ids():
    cb = _cb()
    assert cb.verdict({"W": 1, "F": 1, "R": 1, "B1": 0})
    assert cb.missing({"W": 1, "F": 1, "R": 0, "B1": 1}) == ["productivity without the link"]
    with pytest.raises(ValueError, match="unknown components"):
        _cb().__class__(**{**_cb().to_dict(), "formula": "W & Z"})
    with pytest.raises(ValueError, match="must name"):
        Component("R", "link", kind="relation")


def test_flat_codebook_dict_unchanged():
    flat = Codebook("T1", "x", "d", ["p"])
    assert "components" not in flat.to_dict()


def test_markdown_round_trip():
    cb = _cb(notes="Built by hand.")
    back = from_markdown(to_markdown(cb))
    assert back.to_dict() == {**cb.to_dict(), "frozen": None}
    assert back.content_hash == cb.content_hash


def test_markdown_edit_changes_formula():
    md = to_markdown(_cb()).replace("`W & F & R`", "`W & R`")
    assert from_markdown(md).formula == "W & R"


def test_component_questions_sit_beside_presence():
    cb = _cb()
    t = Topic(topic_id="T1", member_uids=[0], label=cb.label)
    qs = build_question_set([t], codebooks={"T1": cb}, include_depth=False)
    qids = qs.by_topic["T1"]
    assert qids[0] == "T1.presence"
    assert [q for q in qids if role_of(q) == "comp"] == ["T1.comp_W", "T1.comp_F", "T1.comp_R", "T1.comp_B1"]
    r = qs.question("T1.comp_R")
    assert "Names a wellbeing change." in r.text and "separate claims" in r.criteria_false
    assert "office work in general" in qs.question("T1.comp_F").criteria_false


def test_structured_render_marks_examples_and_additions():
    from flashtopic.prevalence.codebook import render
    r = render(_cb())
    assert "not a complete list" in r and "Also met when: implied" in r
    flat = render(Codebook("T1", "x", "d", ["p"], negative_examples=["n"]))
    assert "not a complete list" not in flat  # flat codebooks keep their frozen hashes
