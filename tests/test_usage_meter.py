"""Usage tallies: billed cost parsing and the Jev classifier's per-call record."""

import asyncio

from flashtopic.llm import USAGE, billed_cost_usd, usage_snapshot
from flashtopic.prevalence.classifier import JevClassifier, Question


def test_billed_cost_prefers_upstream_on_byok():
    assert billed_cost_usd({"cost": 0, "is_byok": True,
                            "cost_details": {"upstream_inference_cost": 0.016}}) == 0.016
    assert billed_cost_usd({"cost": 0.002, "is_byok": False}) == 0.002
    assert billed_cost_usd({}) is None
    assert billed_cost_usd(None) is None


def test_jev_call_records_usage():
    q = Question(qid="t.presence", kind="noul", text="Cats?", criteria="Cats.", criteria_false="No.")

    async def transport(payload):
        return {"answers": [{"id": "t.presence", "probabilities": {"yes": 0.9, "no": 0.1}}],
                "usage": {"input_tokens": 300, "output_tokens": 20, "cost": 1e-5}}

    clf = JevClassifier("test/jev", transport=transport)
    before = usage_snapshot().get(clf.name)
    asyncio.run(clf.classify("My cat.", [q]))
    u = USAGE[clf.name]
    assert u.calls == (before.calls if before else 0) + 1
    assert u.input_tokens >= 300 and u.cost_usd >= 1e-5
