"""Codebook entries — the operational "is this document about topic t?" rule.

A codebook makes the membership decision reproducible for the LLM judge and a human
reviewer. :func:`build_codebook` is the deterministic template (label + distinctive
terms + typical exemplars); :mod:`flashtopic.prevalence.draft` has an LLM write richer
ones — include/exclude rules and near-miss negatives — and can split a mixed cluster
into several. Edit the saved JSON before judging if a definition is off; :func:`save` /
:func:`load` round-trip them. Freeze a codebook before drawing its gold sample.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path

from flashtopic.prevalence.topics import Topic


@dataclass
class Codebook:
    topic_id: str
    label: str
    definition: str  # one-line operational definition (editable)
    positive_examples: list[str]
    include: list[str] = field(default_factory=list)  # what counts (drafted rules)
    exclude: list[str] = field(default_factory=list)  # what doesn't, incl. near misses
    negative_examples: list[str] = field(default_factory=list)  # near-miss sentences
    source_topic_id: str | None = None  # the cluster it was drafted from (≠ topic_id on a split)
    notes: str | None = None  # drafter's scope/breadth warnings for the reviewer
    frozen: str | None = None  # content hash, set by freeze(); judging a changed codebook fails

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        """Hash of what the judge is shown (``render``) — the freeze/cache identity."""
        return hashlib.sha256(render(self).encode()).hexdigest()[:16]


def freeze(cb: Codebook) -> Codebook:
    """Pin a codebook before its gold sample is drawn (re-editing after seeing gold labels
    and re-estimating on the same sample would no longer be design-unbiased)."""
    cb.frozen = cb.content_hash
    return cb


def check_frozen(cb: Codebook) -> None:
    """Raise if a frozen codebook was edited after freezing."""
    if cb.frozen and cb.frozen != cb.content_hash:
        raise ValueError(f"{cb.topic_id}: codebook changed after it was frozen — "
                         "draw a new gold sample, or restore the frozen version.")


def build_codebook(topic: Topic, *, corpus_topic: str = "") -> Codebook:
    """Draft a codebook from a labeled topic. Requires a label."""
    if not topic.label:
        raise ValueError(f"{topic.topic_id} has no label — run the labeling stage first.")
    terms = ", ".join(topic.distinctive_terms[:6])
    definition = (
        f"The document substantively discusses {topic.label.lower()}"
        + (f" (themes/terms: {terms})" if terms else "")
        + " — more than a passing mention."
    )
    if topic.definition:
        definition += f" Broader theme: {topic.definition}."
    if corpus_topic:
        definition += f" Context: {corpus_topic}."
    return Codebook(
        topic_id=topic.topic_id,
        label=topic.label,
        definition=definition,
        # Typical members, not the labeler's coverage exemplars: those deliberately span
        # the cluster's range, so as "positive examples" they taught the judge that
        # outliers ("| Area | Needed treatment") count. Coverage is the fallback for
        # topics built before typical exemplars existed.
        positive_examples=list((topic.typical_exemplars or topic.exemplars)[:5]),
    )


def render(cb: Codebook) -> str:
    """Codebook block embedded verbatim in the judge prompt (and shown to a human)."""
    pos = "\n".join(f"  - {t}" for t in cb.positive_examples) or "  (none)"
    out = (
        f"TOPIC: {cb.label}\n"
        f"DEFINITION: {cb.definition}\n"
        f"POSITIVE examples (sentences from on-topic documents):\n{pos}"
    )
    # Drafted sections only when present, so template codebooks render byte-identically
    # (and keep their judge-cache entries).
    if cb.include:
        out += "\nCOUNTS when the document:\n" + "\n".join(f"  - {r}" for r in cb.include)
    if cb.exclude:
        out += "\nDOES NOT COUNT:\n" + "\n".join(f"  - {r}" for r in cb.exclude)
    if cb.negative_examples:
        out += ("\nNEGATIVE examples (near misses — similar wording, NOT this topic):\n"
                + "\n".join(f"  - {t}" for t in cb.negative_examples))
    return out


def save(codebooks: list[Codebook], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.to_dict() for c in codebooks], indent=2))
    return path


def load(path: str | Path) -> list[Codebook]:
    return [Codebook(**d) for d in json.loads(Path(path).read_text())]
