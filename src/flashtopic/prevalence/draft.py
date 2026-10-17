"""LLM codebook drafting — the quality gate between a cluster and a prevalence estimate.

Clusters *suggest* topics; a codebook *defines* one. The template in ``codebook.py`` just
restates the cluster label, which failed in an earlier pilot in three ways: the label
named a concept far broader than the cluster (a "bias" word-cluster labeled as a concept
nearly every response discusses), "more than a passing mention" was undefined for list
items, and a mixed cluster got one codebook for two opposite ideas.

The drafter sees what the cluster actually holds — its *typical* sentences (the core) and
*coverage* sentences (its range), the coherence signals, and the typical sentences of its
nearest neighbouring clusters — and decides:

* ``single`` — one idea: write its codebook, with include/exclude rules;
* ``split``  — two or three distinct ideas: one codebook each (ids ``T7a``, ``T7b``), each
  using the other as a near miss;
* ``narrow`` — the label overstates the cluster: define what the members actually say;
* ``reject`` — boilerplate/noise: no codebook.

Examples are chosen by id from the sentences shown (never written by the model), so a
codebook's examples are always real corpus sentences. The drafter also flags a concept
that is likely present in most documents (a near-universal topic, where prevalence is
uninformative and the membership surrogate will badly undercount).

Drafts are cached by an evidence hash (like atom labels) and are *drafts*: review them —
see ``calibration`` — and :func:`~flashtopic.prevalence.codebook.freeze` before judging.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from flashtopic.llm import call_llm_structured, configure_concurrency
from flashtopic.prevalence.codebook import Codebook
from flashtopic.prevalence.topics import Topic

DRAFT_PROMPT_VERSION = "codebook_draft_v1"
DEFAULT_DRAFT_MODEL = "anthropic/claude-sonnet-5.5"
DEFAULT_PASSING_RULE = (
    "The topic appears only as a bare item in a list, or as a single passing phrase, "
    "with no elaboration of what should be done or why."
)


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #
class DraftedCodebook(BaseModel):
    label: str = Field(description="2-8 word name for exactly this idea.")
    definition: str = Field(
        description="1-2 sentences: the specific claim/idea a document must engage with. "
        "Concrete enough that two coders would agree; not the corpus's overall subject."
    )
    include: list[str] = Field(description="2-4 rules: what a document must do to count.")
    exclude: list[str] = Field(
        description="2-4 rules: what does NOT count — near-miss ideas (name them), and "
        "passing mentions."
    )
    positive_ids: list[str] = Field(
        description="3-5 ids of shown sentences (S#) that clearly express THIS idea."
    )
    negative_ids: list[str] = Field(
        description="2-4 ids of shown sentences (N# from neighbours, or S# members that "
        "belong to a different idea) that look similar but do NOT count."
    )


class CodebookDraft(BaseModel):
    decision: Literal["single", "split", "narrow", "reject"]
    rationale: str = Field(description="1-3 sentences: why this decision, citing sentence ids.")
    label_fits_cluster: bool = Field(
        description="False if the given label names something broader or different than "
        "what the member sentences actually say."
    )
    breadth_warning: str | None = Field(
        default=None,
        description="If the idea is likely present in most documents of this corpus (e.g. "
        "it restates the corpus's own question), say so; else null.",
    )
    codebooks: list[DraftedCodebook] = Field(
        description="1 for single/narrow, 2-3 for split, 0 for reject."
    )


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
@dataclass
class DraftEvidence:
    topic: Topic
    members: list[str]  # typical first, then coverage-only — shown as S1..Sn
    neighbours: list[tuple[str, list[str]]] = field(default_factory=list)  # (label, sentences)
    naive_rate: float | None = None  # membership prevalence, for the breadth judgement

    def neighbour_sentences(self) -> list[str]:
        return [s for _, sents in self.neighbours for s in sents]

    def hash(self, model: str, passing_rule: str, corpus_topic: str) -> str:
        h = hashlib.sha256()
        for part in (DRAFT_PROMPT_VERSION, model, passing_rule, corpus_topic,
                     self.topic.label or "", json.dumps(self.topic.coherence, sort_keys=True),
                     self.topic.label_coherence or "", "|".join(self.topic.discerned_subtopics),
                     "\x1e".join(self.members), json.dumps(self.neighbours),
                     f"{self.naive_rate:.3f}" if self.naive_rate is not None else ""):
            h.update(part.encode() + b"\x1f")
        return h.hexdigest()[:16]


def nearest_topics(topics: list[Topic], E: np.ndarray, n: int = 2) -> dict[str, list[Topic]]:
    """Each topic's ``n`` nearest *other* topics by centroid cosine (the near-miss source)."""
    E = np.asarray(E, dtype=np.float32)
    C = []
    for t in topics:
        c = E[np.asarray(t.member_uids)].mean(axis=0)
        C.append(c / max(float(np.linalg.norm(c)), 1e-12))
    C = np.stack(C)
    S = C @ C.T
    np.fill_diagonal(S, -np.inf)
    return {t.topic_id: [topics[j] for j in np.argsort(-S[i])[:n]] for i, t in enumerate(topics)}


def build_evidence(
    topics: list[Topic], E: np.ndarray, *, n_docs: int | None = None,
    member_doc_counts: dict[str, int] | None = None,
    n_typical: int = 10, n_coverage: int = 8, n_neighbours: int = 2, n_neighbour_sents: int = 4,
) -> dict[str, DraftEvidence]:
    """Assemble drafter evidence for ``topics`` (neighbours are searched among all of them)."""
    near = nearest_topics(topics, E, n_neighbours) if len(topics) > 1 else {}
    out = {}
    for t in topics:
        typ = list(dict.fromkeys(t.typical_exemplars[:n_typical]))
        cov = [s for s in t.exemplars if s not in typ][:n_coverage]
        nb = [(o.label or o.topic_id, (o.typical_exemplars or o.exemplars)[:n_neighbour_sents])
              for o in near.get(t.topic_id, [])]
        rate = None
        if n_docs and member_doc_counts and t.topic_id in member_doc_counts:
            rate = member_doc_counts[t.topic_id] / n_docs
        out[t.topic_id] = DraftEvidence(t, typ + cov, nb, rate)
    return out


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _system_prompt(corpus_topic: str, passing_rule: str) -> str:
    ctx = (f"The corpus: {corpus_topic}. Every document is about that overall subject, so a "
           "codebook must never define a topic at that level.\n\n") if corpus_topic else ""
    return (
        "You write codebooks for measuring topic prevalence: an LLM judge will read whole "
        "documents and decide, using your codebook, whether each one substantively engages "
        "with the topic. Precision matters more than coverage: a vague codebook makes the "
        "judge say yes to almost everything.\n\n"
        + ctx
        + "You are given one cluster of sentences (S#: the first ones are the cluster's core, "
        "the rest show its range), coherence signals, and sentences from the nearest "
        "neighbouring clusters (N#).\n\n"
        "Decide:\n"
        "- single: the members express one idea. Write its codebook.\n"
        "- split: the members hold 2-3 distinct ideas (e.g. opposite positions, or different "
        "subjects sharing vocabulary), each with several members. Write one codebook per idea, "
        "each excluding the others.\n"
        "- narrow: the label names something broader or different from what members say. "
        "Write the codebook for what they actually say.\n"
        "- reject: boilerplate, fragments or noise naming no idea. No codebook.\n\n"
        "Rules for every codebook:\n"
        "- Define the specific idea (a claim, proposal or subject), grounded in the core "
        "sentences, not in shared vocabulary. If members share a word but not an idea, split "
        "or narrow.\n"
        "- Exclude the nearest neighbouring ideas by name, and always include this passing-"
        f"mention rule in exclude: \"{passing_rule}\"\n"
        "- Choose examples only by id from the sentences shown. Positive: clear instances. "
        "Negative: similar-looking sentences that do not count.\n"
        "- If the idea likely appears in most documents of this corpus, set breadth_warning."
    )


def _user_prompt(ev: DraftEvidence) -> str:
    t = ev.topic
    coh = t.coherence or {}
    lines = [
        f"CLUSTER {t.topic_id} — current label: {t.label or '(none)'}",
        f"Distinctive terms: {', '.join(t.distinctive_terms[:10])}",
        (f"Coherence (cosine to centroid): median {coh.get('median', float('nan')):.2f}, "
         f"10th percentile {coh.get('p10', float('nan')):.2f}  (typical clusters here: ~0.55 / ~0.40)")
        if coh else "Coherence: unknown",
        f"Labeler's verdict: {t.label_coherence or 'unknown'}"
        + (f" — facets: {'; '.join(t.discerned_subtopics)}" if t.discerned_subtopics else ""),
    ]
    if ev.naive_rate is not None:
        lines.append(f"Share of documents with a member sentence: {ev.naive_rate:.0%}")
    n_typ = len(dict.fromkeys(t.typical_exemplars))
    lines.append("\nMember sentences (S1.. are the core; later ones show the range):")
    lines += [f"S{i} {'(core) ' if i <= n_typ else ''}{s}" for i, s in enumerate(ev.members, 1)]
    if ev.neighbours:
        lines.append("\nNearest neighbouring clusters:")
        k = 0
        for label, sents in ev.neighbours:
            lines.append(f"  [{label}]")
            for s in sents:
                k += 1
                lines.append(f"  N{k} {s}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Drafting
# --------------------------------------------------------------------------- #
@dataclass
class DraftResult:
    topic_id: str
    decision: str
    rationale: str
    label_fits_cluster: bool
    breadth_warning: str | None
    codebooks: list[Codebook]
    cost_usd: float = 0.0
    error: str | None = None


def _resolve(ids: list[str], ev: DraftEvidence, allowed: str) -> list[str]:
    """Map S#/N# ids to sentence text; drop ids that don't exist or aren't ``allowed``."""
    pools = {"S": ev.members, "N": ev.neighbour_sentences()}
    out = []
    for raw in ids:
        tag, num = raw.strip()[:1].upper(), raw.strip()[1:]
        if tag in allowed and num.isdigit() and 1 <= int(num) <= len(pools[tag]):
            s = pools[tag][int(num) - 1]
            if s not in out:
                out.append(s)
    return out


def to_result(ev: DraftEvidence, draft: CodebookDraft | None, *, cost: float = 0.0,
              error: str | None = None) -> DraftResult:
    tid = ev.topic.topic_id
    if draft is None:
        return DraftResult(tid, "error", error or "no draft", False, None, [], cost, error)
    cbs = []
    drafted = draft.codebooks if draft.decision != "reject" else []
    for i, d in enumerate(drafted):
        cid = tid if len(drafted) == 1 else f"{tid}{chr(ord('a') + i)}"
        notes = " ".join(x for x in (
            None if draft.label_fits_cluster else f"Label '{ev.topic.label}' did not fit the cluster.",
            f"Breadth: {draft.breadth_warning}" if draft.breadth_warning else None,
            f"Drafter ({draft.decision}): {draft.rationale}",
        ) if x)
        cbs.append(Codebook(
            topic_id=cid, label=d.label, definition=d.definition,
            positive_examples=_resolve(d.positive_ids, ev, "S")[:5],
            include=list(d.include), exclude=list(d.exclude),
            negative_examples=_resolve(d.negative_ids, ev, "SN")[:4],
            source_topic_id=tid, notes=notes,
        ))
    return DraftResult(tid, draft.decision, draft.rationale, draft.label_fits_cluster,
                       draft.breadth_warning, cbs, cost)


async def draft_codebook(ev: DraftEvidence, *, model: str = DEFAULT_DRAFT_MODEL,
                         corpus_topic: str = "", passing_rule: str = DEFAULT_PASSING_RULE
                         ) -> DraftResult:
    parsed, raw = await call_llm_structured(
        _system_prompt(corpus_topic, passing_rule), _user_prompt(ev), model, CodebookDraft,
        temperature=0.2, max_tokens=3000,
    )
    return to_result(ev, parsed, cost=raw.cost_usd, error=raw.error)


async def draft_codebooks(
    evidence: dict[str, DraftEvidence], *, model: str = DEFAULT_DRAFT_MODEL,
    corpus_topic: str = "", passing_rule: str = DEFAULT_PASSING_RULE,
    concurrency: int = 8, cache_path: str | Path | None = None,
) -> dict[str, DraftResult]:
    """Draft every topic's codebook(s); reuse cached drafts on an evidence-hash match."""
    configure_concurrency(concurrency)
    cache: dict = {}
    if cache_path and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())

    async def _one(tid: str, ev: DraftEvidence):
        h = ev.hash(model, passing_rule, corpus_topic)
        hit = cache.get(tid)
        if hit and hit.get("hash") == h:
            return tid, to_result(ev, CodebookDraft.model_validate(hit["draft"]))
        parsed, raw = await call_llm_structured(
            _system_prompt(corpus_topic, passing_rule), _user_prompt(ev), model, CodebookDraft,
            temperature=0.2, max_tokens=3000,
        )
        if parsed is not None:
            cache[tid] = {"hash": h, "draft": parsed.model_dump()}
        return tid, to_result(ev, parsed, cost=raw.cost_usd, error=raw.error)

    pairs = await asyncio.gather(*(_one(t, e) for t, e in evidence.items()))
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache, indent=1))
    return dict(pairs)


def run_drafts(evidence, **kw) -> dict[str, DraftResult]:
    """Synchronous wrapper around :func:`draft_codebooks`."""
    return asyncio.run(draft_codebooks(evidence, **kw))


def topic_for_codebook(parent: Topic, cb: Codebook) -> Topic:
    """A Topic carrying a (possibly split) codebook's id and label, sharing the parent's
    members — prevalence is measured against the codebook; membership only guides
    sampling, and the estimator stays unbiased whatever it is."""
    return replace(parent, topic_id=cb.topic_id, label=cb.label)
