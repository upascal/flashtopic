"""Classifier adapter: protocol, mock, request/response round-trip, cache semantics."""

from __future__ import annotations

import asyncio
import json

import pytest

from flashtopic.prevalence.classifier import (
    Answer, DocClassifier, JevClassifier, MockClassifier, Question, build_request,
    classify_doc, classify_docs, decisions_url, parse_answers,
)

Q = [
    Question("T1.presence", "noul", "Does it discuss X?", criteria="Definition: X"),
    Question("T1.depth", "score", "How much?", options=("none", "some", "lots")),
    Question("group:1.sibling", "choice", "Which?", options=("a", "b", "none of these")),
]


def test_question_validation():
    with pytest.raises(ValueError):
        Question("q", "bogus", "x")
    with pytest.raises(ValueError):
        Question("q", "choice", "x", options=("only",))
    with pytest.raises(ValueError):
        Question("q", "score", "x", options=tuple(str(i) for i in range(11)))
    assert Q[0].content_hash != Question("other.id", "noul", "Does it discuss Y?").content_hash
    assert Q[0].content_hash == Question("other.id", "noul", "Does it discuss X?", criteria="Definition: X").content_hash
    assert Q[0].content_hash != Question("T1.presence", "noul", "Does it discuss X?", criteria="Definition: X",
                                         criteria_false="only in passing").content_hash
    with pytest.raises(ValueError):
        Question("q", "score", "x", options=("a", "b"), criteria_false="nope")


def test_mock_satisfies_protocol_and_is_deterministic():
    m = MockClassifier()
    assert isinstance(m, DocClassifier)
    a = asyncio.run(m.classify("a doc about discuss discuss", Q))
    b = asyncio.run(m.classify("a doc about discuss discuss", Q))
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert a[0].p_yes is not None and len(a[1].dist) == 3 and len(a[2].dist) == 3
    assert abs(sum(a[2].dist) - 1) < 1e-9


def test_decisions_url_derivation():
    assert decisions_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/alpha/decisions"
    assert decisions_url("https://example.com/") == "https://example.com/api/alpha/decisions"


def test_build_request_and_parse_round_trip():
    req = build_request("typesafe/jev-1.13", "doc text", Q)
    assert req["model"] == "typesafe/jev-1.13" and req["state"] == "doc text"
    pres = req["questions"]["T1.presence"]
    assert pres["type"] == "noul" and pres["instructions"] == "Does it discuss X?"
    assert pres["criteria"]["true"] == "Definition: X" and "false" in pres["criteria"]
    neg = Question("q", "noul", "X?", criteria="yes side", criteria_false="no side").to_wire()
    assert neg["criteria"] == {"true": "yes side", "false": "no side"}
    assert req["questions"]["T1.depth"]["criteria"] == ["none", "some", "lots"]
    assert req["questions"]["group:1.sibling"]["criteria"] == {
        "a": "a", "b": "b", "none of these": "none of these"}

    resp = {"answers": {
        "T1.presence": {"type": "noul", "noul": 0.83},
        "T1.depth": {"type": "score", "score": 1.1, "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}, "confidence": 0.7},
        "group:1.sibling": {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.7, "none of these": 0.1}},
    }}
    ans = parse_answers(Q, resp)
    assert ans[0].p_yes == pytest.approx(0.83)
    assert ans[1].dist == pytest.approx((0.1, 0.7, 0.2)) and ans[1].confidence == 0.7
    assert ans[2].dist == pytest.approx((0.2, 0.7, 0.1))
    assert all(a.ok for a in ans)

    # tolerant of a flat top-level map and a missing question
    flat = parse_answers(Q, {"T1.presence": {"noul": 0.5}})
    assert flat[0].ok and not flat[1].ok and flat[1].error == "missing"


def test_jev_transport_injection_retries_then_reports_error():
    calls = {"n": 0}

    async def flaky(payload):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("boom")
        return {"answers": {q: {"noul": 0.9, "probabilities": {"none": 0.1, "some": 0.2, "lots": 0.7,
                                                                 "a": 0.5, "b": 0.3, "none of these": 0.2}}
                            for q in payload["questions"]}}

    import flashtopic.prevalence.classifier as mod
    mod._backoff = lambda attempt: 0.0  # no sleeping in tests
    clf = JevClassifier("typesafe/jev-1.13", transport=flaky, max_questions_per_request=2)
    ans = asyncio.run(clf.classify("doc", Q))
    assert clf.name == "jev:typesafe/jev-1.13"
    assert [a.ok for a in ans] == [True, True, True]
    assert calls["n"] >= 2  # 2 chunks (2 + 1 questions), first attempt failed

    async def dead(payload):
        raise RuntimeError("down")

    clf = JevClassifier(transport=dead, max_attempts=2)
    ans = asyncio.run(clf.classify("doc", Q[:1]))
    assert not ans[0].ok and "down" in ans[0].error


def test_cache_hits_skip_calls_and_never_store_errors(tmp_path):
    def rule(doc, q):
        if q.qid == "T1.depth":
            return Answer(q.qid, error="nope")
        return Answer(q.qid, p_yes=0.7) if q.kind == "noul" else Answer(q.qid, dist=(0.2, 0.3, 0.5))

    m = MockClassifier(rule)
    a1 = asyncio.run(classify_doc(m, 7, "text", Q, cache_dir=tmp_path))
    assert m.calls == 1 and m.questions_seen == 3
    assert a1[0].ok and not a1[1].ok
    cached = json.loads((tmp_path / "mock" / "7.json").read_text())
    assert len(cached) == 2  # the error was not cached

    a2 = asyncio.run(classify_doc(m, 7, "text", Q, cache_dir=tmp_path))
    assert m.calls == 2 and m.questions_seen == 4  # only the failed question was re-asked
    assert a2[0].qid == "T1.presence" and a2[0].p_yes == 0.7

    # a renamed qid with identical content is a cache hit; changed content is not
    renamed = [Question("T2.presence", "noul", "Does it discuss X?", criteria="Definition: X")]
    asyncio.run(classify_doc(m, 7, "text", renamed, cache_dir=tmp_path))
    assert m.questions_seen == 4
    changed = [Question("T2.presence", "noul", "Does it discuss Z?")]
    asyncio.run(classify_doc(m, 7, "text", changed, cache_dir=tmp_path))
    assert m.questions_seen == 5

    out = asyncio.run(classify_docs(m, {7: "text", 8: "other"}, Q[:1], cache_dir=tmp_path, concurrency=2))
    assert set(out) == {7, 8} and out[7][0].p_yes == 0.7
