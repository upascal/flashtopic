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
from pydantic import BaseModel, Field, create_model

from flashtopic.llm import call_llm_structured, configure_concurrency
from flashtopic.prevalence.codebook import Codebook, render, render_component
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


# --------------------------------------------------------------------------- #
# Structured codebooks: a point-blank verdict plus one finding per component
# --------------------------------------------------------------------------- #
COMPONENT_PROMPT_VERSION = "c1"

COMPONENT_SYS_PROMPT = (
    "You are a careful annotator measuring topic prevalence across a document corpus. "
    "Given a TOPIC codebook and one full document, first answer point blank: does the "
    "document substantively engage with the topic as defined? Then check each listed "
    "CONDITION on its own, quoting the deciding words. A relation condition is met only "
    "when the conditions it links hold of the same claim and connect as stated — both "
    "appearing somewhere in the document is not enough. Judge what the author proposes, "
    "not views the author reports or argues against, unless a condition says otherwise."
)


class ComponentFinding(BaseModel):
    id: str = Field(description="The condition id, e.g. W or R.")
    met: bool = Field(description="Does the document meet this condition?")
    evidence: str = Field(description="Short quote that decides it, or '' if not met.")


class ComponentJudgeVerdict(BaseModel):
    discusses: bool = Field(description="Point blank: does the document substantively engage "
                                        "with the topic as defined?")
    confidence: float = Field(description="Confidence in that verdict, between 0 and 1.")
    findings: list[ComponentFinding] = Field(description="One finding per listed condition.")
    rationale: str = Field(description="One sentence on the deciding evidence.")


COMPONENT_SCHEMAS = ("list", "fields")


def component_fields_model(cb: Codebook) -> type[BaseModel]:
    """The "fields" schema: one met/evidence pair per condition, each field described by its
    condition and edge cases, so the rule travels with the field it decides as well as in
    the prompt. Same order as the list schema: point blank first, rationale last."""
    fields: dict = {
        "discusses": (bool, ComponentJudgeVerdict.model_fields["discusses"]),
        "confidence": (float, ComponentJudgeVerdict.model_fields["confidence"]),
    }
    for c in cb.components:
        fields[f"{c.id}_met"] = (bool, Field(description=render_component(c)))
        fields[f"{c.id}_evidence"] = (str, Field(description=f"Short quote that decides [{c.id}], "
                                                             "or '' if not met."))
    fields["rationale"] = (str, ComponentJudgeVerdict.model_fields["rationale"])
    return create_model(f"{cb.topic_id}ConditionVerdict", **fields)


@dataclass
class ComponentVerdict:
    doc_id: int
    discusses: bool  # the point-blank answer
    confidence: float
    met: dict[str, bool]  # component id → met
    evidence: dict[str, str]
    rationale: str
    cost_usd: float
    cached: bool
    error: str | None = None

    def formula_verdict(self, cb: Codebook) -> bool:
        return cb.verdict(self.met)


def _component_cache_path(cb: Codebook, doc_id: int, model: str, cache_dir: Path,
                          schema: str = "list") -> Path:
    # The codebook hash is in the key: a structured codebook is edited in pieces, and a stale
    # finding for a changed condition would silently mix two codebooks.
    key = f"{cb.topic_id}|{doc_id}|{COMPONENT_PROMPT_VERSION}|{model}|{cb.content_hash}"
    if schema != "list":  # the list schema keeps its original keys
        key += f"|{schema}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{cb.topic_id}_components" / f"{doc_id}_{h}.json"


def _component_verdict(doc_id: int, d: dict, cost: float, cached: bool) -> ComponentVerdict:
    return ComponentVerdict(doc_id, d["discusses"], d["confidence"],
                            {f["id"]: f["met"] for f in d["findings"]},
                            {f["id"]: f["evidence"] for f in d["findings"]},
                            d["rationale"], cost, cached)


async def judge_components(
    cb: Codebook, doc_id: int, doc_text: str, model: str,
    *, cache_dir: Path = _DEFAULT_CACHE_DIR, schema: str = "list",
) -> ComponentVerdict:
    """One cached structured-judge call; findings for every component or an error.

    ``schema="list"`` asks for a generic list of findings; ``"fields"`` gives each condition
    its own described fields (see :func:`component_fields_model`). The prompt is the same.
    """
    if not cb.structured:
        raise ValueError(f"{cb.topic_id}: judge_components needs a structured codebook")
    if schema not in COMPONENT_SCHEMAS:
        raise ValueError(f"schema must be one of {COMPONENT_SCHEMAS}, got {schema!r}")
    path = _component_cache_path(cb, doc_id, model, cache_dir, schema)
    if path.exists():
        return _component_verdict(doc_id, json.loads(path.read_text()), 0.0, True)
    user = (f"{render(cb)}\n\n=== DOCUMENT ===\n{doc_text[:_DOC_CHAR_CAP]}\n=== END ===\n\n"
            f"Answer point blank, then give a finding for each condition: "
            f"{', '.join(c.id for c in cb.components)}.")
    response_model = ComponentJudgeVerdict if schema == "list" else component_fields_model(cb)
    parsed, res = await call_llm_structured(COMPONENT_SYS_PROMPT, user, model, response_model,
                                            temperature=0.0, max_tokens=1500)
    if parsed is not None and schema == "fields":  # back to the list shape the cache stores
        f = parsed.model_dump()
        parsed = ComponentJudgeVerdict(
            discusses=f["discusses"], confidence=f["confidence"], rationale=f["rationale"],
            findings=[ComponentFinding(id=c.id, met=f[f"{c.id}_met"], evidence=f[f"{c.id}_evidence"])
                      for c in cb.components])
    ids = {c.id for c in cb.components}
    if parsed is None or {f.id for f in parsed.findings} != ids:  # don't cache: retry later
        err = res.error or ("no verdict" if parsed is None else "findings don't match the conditions")
        return ComponentVerdict(doc_id, False, 0.0, {}, {}, f"judge failed: {err}",
                                res.cost_usd, False, err)
    d = parsed.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))
    return _component_verdict(doc_id, d, res.cost_usd, False)


async def judge_components_docs(cb: Codebook, doc_texts: dict[int, str], model: str,
                                *, concurrency: int = 8, cache_dir: Path = _DEFAULT_CACHE_DIR,
                                schema: str = "list") -> list[ComponentVerdict]:
    configure_concurrency(concurrency)
    return await asyncio.gather(*(judge_components(cb, int(d), t, model, cache_dir=cache_dir,
                                                   schema=schema)
                                  for d, t in doc_texts.items()))
