"""LLM cluster labeling: per-cluster (atom) + contrastive (sibling-differentiating).

* **Atom labeling** names each cluster in isolation (meta-theme + specific topic +
  a coherence verdict that flags covertly-mixed clusters).
* **Contrastive labeling** names a *set* of related clusters in one call, asking the
  LLM what makes each DISTINCT — turning near-duplicate labels ("Curie's Nobel prizes"
  × N) into differentiated ones. For a flat k-means output the "siblings" are the
  clusters themselves; oversized sets are batched by cosine similarity so the
  most-confusable land in the same call.

An optional ``corpus_topic`` (the overarching subject, *if* the corpus has one) comes
from the caller (Workspace). When given, prompts steer labels *below* that level; when
omitted — the default for an arbitrary corpus — the prompts read as plain general-corpus
labeling, never interpolating an empty topic. Distinctive terms are computed upstream
(``terms.py``) and passed in; this module only does the LLM call, batching, and caching.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from flashtopic.llm import LLMResult, call_llm_structured, configure_concurrency

ATOM_PROMPT_VERSION = "flashtopic_atom_v2"
CONTRASTIVE_PROMPT_VERSION = "flashtopic_contrastive_v2"


# =========================================================================== #
# Atom labeling (one cluster at a time)
# =========================================================================== #
class AtomLabel(BaseModel):
    meta_theme: str = Field(description="Broader category grouping this cluster with related ones. 2-6 word noun phrase.")
    specific_topic: str = Field(description="The specific facet these sentences are about. 2-6 word noun phrase.")
    similarity: str = Field(description="One sentence: what unifies these exemplars.")
    differences: str | None = Field(default=None, description="One sentence on meaningful variation, or null.")
    coherence: Literal["coherent", "mixed", "noise"] = Field(
        description="'coherent': one topic dominates. 'mixed': ≥2 different topics. 'noise': boilerplate naming no topic."
    )
    discerned_subtopics: list[str] = Field(
        default_factory=list, description="If 'mixed', the distinct facets as 2-6 word noun phrases; else empty."
    )


@dataclass
class AtomEvidence:
    atom_id: int
    terms: list[str]
    exemplars: list[str]
    exemplar_rare_terms: list[list[str]] | None = None


def _atom_system_prompt(corpus_topic: str = "") -> str:
    if corpus_topic:
        intro = (
            f"You are labeling one cluster of sentences from a corpus about {corpus_topic}. "
            "Because every cluster is trivially 'about' that overall topic, do NOT label at "
            "that level. "
        )
    else:
        intro = "You are labeling one cluster of sentences from a text corpus. "
    return (
        intro
        + "Name (a) the SPECIFIC topic the cluster is about, and (b) the broader META-THEME "
        "it fits under -- something that groups it with related but distinct clusters.\n\n"
        "Articulate similarity AND difference. Flag 'mixed' only when the differences are "
        "substantial enough to split the cluster. Flag 'noise' for boilerplate/generic text that "
        "names no topic (intro/dateline sentences, structural summaries)."
    )


def _atom_user_prompt(terms, exemplars, max_chunk_chars, exemplar_rare_terms=None) -> str:
    lines = []
    for i, c in enumerate(exemplars, 1):
        line = f"[{i}] {c[:max_chunk_chars].replace(chr(10), ' ').strip()}"
        if exemplar_rare_terms is not None and i - 1 < len(exemplar_rare_terms):
            rare = exemplar_rare_terms[i - 1][:3]
            if rare:
                line += f"\n     (rarest words: {', '.join(rare)})"
        lines.append(line)
    return (
        "Distinctive terms for this cluster vs all other clusters:\n" + ", ".join(terms) + "\n\n"
        + f"{len(exemplars)} exemplar sentences spanning the cluster's spread (NOT just the core "
        "-- includes the periphery so you can judge whether it is genuinely one topic or covertly "
        "mixed):\n" + "\n".join(lines)
    )


def evidence_hash(ev: AtomEvidence, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(ATOM_PROMPT_VERSION.encode())
    h.update("\x1f".join(ev.terms).encode())
    h.update("\x1f".join(ev.exemplars).encode())
    if ev.exemplar_rare_terms:
        h.update("\x1f".join("|".join(x) for x in ev.exemplar_rare_terms).encode())
    return h.hexdigest()[:16]


async def label_atom(ev: AtomEvidence, *, model: str, corpus_topic: str = "",
                     max_chunk_chars: int = 300) -> tuple[AtomLabel | None, LLMResult]:
    return await call_llm_structured(
        _atom_system_prompt(corpus_topic),
        _atom_user_prompt(ev.terms, ev.exemplars, max_chunk_chars, ev.exemplar_rare_terms),
        model, AtomLabel, temperature=0.2, max_tokens=2000,
    )


async def label_atoms_all(
    evidence: dict[int, AtomEvidence], *, model: str, corpus_topic: str = "",
    max_chunk_chars: int = 300, concurrency: int = 8, cache: dict | None = None,
) -> dict[int, dict]:
    """Label every cluster; reuse cached entries on evidence-hash match."""
    configure_concurrency(concurrency)
    cache = cache if cache is not None else {}

    async def _run(atom_id, ev):
        h = evidence_hash(ev, model)
        existing = cache.get(str(atom_id))
        if existing and existing.get("evidence_hash") == h and not existing.get("raw_error"):
            return atom_id, existing
        parsed, raw = await label_atom(ev, model=model, corpus_topic=corpus_topic,
                                       max_chunk_chars=max_chunk_chars)
        entry = {
            "atom_id": atom_id,
            **(parsed.model_dump() if parsed else {
                "meta_theme": None, "specific_topic": None, "similarity": None,
                "differences": None, "coherence": "error", "discerned_subtopics": []}),
            "evidence_hash": h, "model": model, "raw_error": raw.error,
            "cost_usd": raw.cost_usd, "input_tokens": raw.input_tokens, "output_tokens": raw.output_tokens,
        }
        return atom_id, entry

    results = await asyncio.gather(*[_run(aid, ev) for aid, ev in evidence.items()])
    out = {}
    for aid, entry in results:
        cache[str(aid)] = entry
        out[aid] = entry
    return out


# =========================================================================== #
# Contrastive labeling (a set of related clusters in one call)
# =========================================================================== #
class ContrastiveChild(BaseModel):
    ref: str = Field(description="The exact ref id given for this child; copy it back verbatim.")
    label: str = Field(description="3-8 word noun phrase: enough shared subject to stand alone PLUS the distinguishing facet. Not a bare differentiator or vacuous comparative.")
    differs: str = Field(description="One sentence: how this child differs from the others in the set.")


class ContrastiveGroup(BaseModel):
    parent_theme: str = Field(description="2-6 word noun phrase naming the shared theme uniting these siblings.")
    parent_unifies: str = Field(description="One sentence: what these siblings have in common.")
    children: list[ContrastiveChild] = Field(description="One entry per child, each echoing its ref id.")


@dataclass
class ChildEvidence:
    ref: str
    terms: list[str]
    evidence_lines: list[str]
    subject_terms: list[str] = field(default_factory=list)


def _group_system_prompt(child_level: str, parent_level: str, corpus_topic: str = "") -> str:
    about = f" over a corpus about {corpus_topic}" if corpus_topic else ""
    return (
        f"You are labeling the {child_level}s within one {parent_level} of a topic structure{about}. "
        "They are closely related -- an earlier pass labeled many of "
        "them almost identically.\n\n"
        f"For each {child_level}, write a label balancing what it SHARES with its siblings and what "
        "SETS IT APART: name the concrete shared subject so the label stands on its own, then pin "
        "down the specific facet that distinguishes it.\n\n"
        "Avoid two failure modes: (1) labels so generic they only restate the shared theme; (2) "
        "labels so purely contrastive they become vacuous comparatives ('most iconic', 'standout'). "
        f"Also name the shared {parent_level} theme. Echo each child's ref id back verbatim."
    )


def _group_user_prompt(children, neighbor_terms, *, child_level, max_lines, max_chunk_chars) -> str:
    blocks = []
    for c in children:
        header = f"[{c.ref}]"
        if c.subject_terms:
            header += " shared-subject terms: " + ", ".join(c.subject_terms)
        header += " | distinguishing terms: " + (", ".join(c.terms) if c.terms else "(none)")
        lines = [header] + [f"   - {ev[:max_chunk_chars].replace(chr(10), ' ').strip()}"
                            for ev in c.evidence_lines[:max_lines]]
        blocks.append("\n".join(lines))
    out = f"The {child_level}s to differentiate:\n\n" + "\n\n".join(blocks)
    if neighbor_terms:
        nb = "; ".join(", ".join(t) for t in neighbor_terms if t)
        if nb:
            out += (f"\n\nOther {child_level}s in this same group you are NOT labeling here -- keep "
                    "your labels distinct from these too:\n" + nb)
    return out


async def label_group(
    children: list[ChildEvidence], *, model: str, corpus_topic: str = "", child_level: str,
    parent_level: str, neighbor_terms: list[list[str]] | None = None,
    max_lines: int = 5, max_chunk_chars: int = 300, max_tokens: int = 3000,
) -> tuple[ContrastiveGroup | None, LLMResult]:
    return await call_llm_structured(
        _group_system_prompt(child_level, parent_level, corpus_topic),
        _group_user_prompt(children, neighbor_terms, child_level=child_level,
                           max_lines=max_lines, max_chunk_chars=max_chunk_chars),
        model, ContrastiveGroup, temperature=0.2, max_tokens=max_tokens,
    )


def _group_hash(children, neighbor_terms, model) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(CONTRASTIVE_PROMPT_VERSION.encode())
    for c in children:
        h.update(c.ref.encode())
        h.update("\x1f".join(c.terms).encode())
        h.update("\x1f".join(c.subject_terms).encode())
        h.update("\x1f".join(c.evidence_lines).encode())
    for t in neighbor_terms or []:
        h.update("\x1e".join(t).encode())
    return h.hexdigest()[:16]


async def label_level_all(
    calls: dict[str, dict], *, model: str, corpus_topic: str = "", child_level: str, parent_level: str,
    concurrency: int = 8, cache: dict | None = None, max_lines: int = 5, max_chunk_chars: int = 300,
) -> dict[str, dict]:
    """One contrastive call per entry in `calls` (each may be one batch of a larger set).

    ``calls[key] = {"children": list[ChildEvidence], "neighbor_terms": [[str]] | None}``.
    """
    configure_concurrency(concurrency)
    cache = cache if cache is not None else {}

    async def _run(key, spec):
        children = spec["children"]
        neighbor_terms = spec.get("neighbor_terms")
        h = _group_hash(children, neighbor_terms, model)
        existing = cache.get(key)
        if existing and existing.get("evidence_hash") == h and not existing.get("raw_error"):
            return key, existing
        parsed, raw = await label_group(children, model=model, corpus_topic=corpus_topic,
                                        child_level=child_level, parent_level=parent_level,
                                        neighbor_terms=neighbor_terms, max_lines=max_lines,
                                        max_chunk_chars=max_chunk_chars)
        if parsed:
            entry = {"parent_theme": parsed.parent_theme, "parent_unifies": parsed.parent_unifies,
                     "children": {c.ref: {"label": c.label, "differs": c.differs} for c in parsed.children}}
        else:
            entry = {"parent_theme": None, "parent_unifies": None, "children": {}}
        entry.update(evidence_hash=h, model=model, raw_error=raw.error, cost_usd=raw.cost_usd,
                     input_tokens=raw.input_tokens, output_tokens=raw.output_tokens)
        return key, entry

    results = await asyncio.gather(*[_run(k, s) for k, s in calls.items()])
    out = {}
    for key, entry in results:
        cache[key] = entry
        out[key] = entry
    return out


def batch_by_similarity(child_ids: list[str], child_medoids: np.ndarray, *, n: int) -> list[list[str]]:
    """Split `child_ids` into similarity-coherent batches of size <= n (cosine agglomeration).

    `child_medoids` is (len(child_ids), d) L2-normalized. Most-confusable children land in
    the same call. <= n children → a single batch.
    """
    m = len(child_ids)
    if m <= n:
        return [list(child_ids)]
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    order = leaves_list(linkage(pdist(child_medoids, metric="cosine"), method="average"))
    k_batches = -(-m // n)
    return [[child_ids[int(i)] for i in chunk] for chunk in np.array_split(order, k_batches)]
