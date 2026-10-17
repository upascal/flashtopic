"""LLM-as-judge document labeling — the gold labels (Y) for a sampled design.

For each sampled document the judge decides, against the codebook, whether the
*document* substantively discusses the topic (not just mentions it in passing). The
verdict's ``discusses`` is the gold label the estimator debiases the cheap surrogate
against. Calls route through ``flashtopic.llm`` (OpenRouter), and every verdict is
cached on disk by ``(topic, doc, prompt_version, model)`` so reruns and the human
spot-check are free.

This is "LLM-as-gold": cheaper than human annotation, but itself a surrogate — validate
it against a small **human** spot-check before trusting any number.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from flashtopic.llm import call_llm_structured, configure_concurrency
from flashtopic.prevalence.codebook import Codebook, render
from flashtopic.prevalence.sampling import SampleDesign

PROMPT_VERSION = "v2"
_DEFAULT_CACHE_DIR = Path.cwd() / ".flashtopic_cache" / "judge"
_DOC_CHAR_CAP = 16000  # comfortably holds a full document

SYS_PROMPT = (
    "You are a careful annotator measuring topic prevalence across a document corpus. "
    "Given a TOPIC codebook and one full document, decide whether the document "
    "SUBSTANTIVELY discusses the topic — it devotes a clause, sentence, or more to it, "
    "not merely a passing name-drop or incidental mention. Apply the codebook's "
    "definition; the positive examples show what counts. When the topic is only alluded "
    "to in passing, answer false."
)


class JudgeVerdict(BaseModel):
    # NOTE: no ge/le on confidence — Anthropic structured-output mode rejects
    # minimum/maximum on number types. Bound it in the description instead.
    discusses: bool = Field(description="Does the document substantively discuss the topic?")
    confidence: float = Field(description="Confidence between 0 and 1.")
    rationale: str = Field(description="One sentence citing the deciding evidence.")


@dataclass
class Verdict:
    doc_id: int
    discusses: bool
    confidence: float
    rationale: str
    cost_usd: float
    cached: bool


def _cache_path(topic_id: str, doc_id: int, model: str, cache_dir: Path) -> Path:
    key = f"{topic_id}|{doc_id}|{PROMPT_VERSION}|{model}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Path(cache_dir) / topic_id / f"{doc_id}_{h}.json"


def _user_prompt(cb_text: str, doc_text: str) -> str:
    return (
        f"{cb_text}\n\n"
        f"=== DOCUMENT ===\n{doc_text[:_DOC_CHAR_CAP]}\n=== END ===\n\n"
        "Does this document substantively discuss the TOPIC above?"
    )


async def judge_doc(
    cb: Codebook, doc_id: int, doc_text: str, model: str, cb_text: str | None = None,
    *, cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> Verdict:
    """One cached judge call. Returns the cached verdict without an API call on a hit."""
    path = _cache_path(cb.topic_id, doc_id, model, cache_dir)
    if path.exists():
        d = json.loads(path.read_text())
        return Verdict(doc_id, d["discusses"], d["confidence"], d["rationale"], 0.0, True)

    cb_text = cb_text if cb_text is not None else render(cb)
    parsed, res = await call_llm_structured(
        SYS_PROMPT, _user_prompt(cb_text, doc_text), model, JudgeVerdict,
        temperature=0.0, max_tokens=300,
    )
    if parsed is None:  # parse/API failure after retries — do NOT cache, so it retries
        return Verdict(doc_id, False, 0.0, f"judge failed: {res.error or 'no verdict'}",
                       res.cost_usd, False)
    v = Verdict(doc_id, parsed.discusses, parsed.confidence, parsed.rationale, res.cost_usd, False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"discusses": v.discusses, "confidence": v.confidence,
                                "rationale": v.rationale}))
    return v


async def judge_design(
    cb: Codebook,
    design: SampleDesign,
    doc_texts: dict[int, str],
    model: str = "anthropic/claude-haiku-4.5",
    concurrency: int = 8,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> list[Verdict]:
    """Judge every document in a sample design (order matches ``design.doc_ids``)."""
    configure_concurrency(concurrency)
    cb_text = render(cb)
    tasks = [
        judge_doc(cb, int(d), doc_texts[int(d)], model, cb_text, cache_dir=cache_dir)
        for d in design.doc_ids
    ]
    return await asyncio.gather(*tasks)


def run_judge(cb, design, doc_texts, **kw) -> list[Verdict]:
    """Synchronous wrapper around :func:`judge_design` for cell-mode driving."""
    return asyncio.run(judge_design(cb, design, doc_texts, **kw))


def verdicts_to_y(design: SampleDesign, verdicts: list[Verdict]) -> np.ndarray:
    """Align verdicts to ``design.doc_ids`` order → gold label array Y (1/0)."""
    by_doc = {v.doc_id: int(v.discusses) for v in verdicts}
    return np.array([by_doc[int(d)] for d in design.doc_ids], dtype=int)
