"""Codebook entries — the operational "is this document about topic t?" rule.

A codebook makes the membership decision reproducible for the LLM judge and a human
reviewer. :func:`build_codebook` is the deterministic template (label + distinctive
terms + typical exemplars); :mod:`flashtopic.prevalence.draft` has an LLM write richer
ones — include/exclude rules and near-miss negatives — and can split a mixed cluster
into several. Edit the saved JSON before judging if a definition is off; :func:`save` /
:func:`load` round-trip them. Freeze a codebook before drawing its gold sample.

A codebook can also be **structured**: ``components`` are single checkable conditions
(``P`` names a policy, ``O`` names an outcome …), some of them *relations* that say how
others must connect (``R``: P is presented as the cause of O, in the same claim), and
``formula`` is the boolean rule over their ids that decides membership (``P & O & R``).
Exclusions become ``near_misses``: named formulas saying which piece is missing
(``(B1 | B2) & ~R`` — both mentioned without the link). Each component is
asked separately, so a disagreement points at one condition rather than at a whole rule.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

from flashtopic.prevalence.topics import Topic

COMPONENT_KINDS = ("atom", "relation")
# Edge-case labels. "Also met" makes a yes_when an addition to the condition, not its only
# case ("met when: the outcome is implied" read as "only when implied").
ALSO_MET = "Also met when"
NOT_MET = "Not met when"
POS_EXAMPLES_HEAD = ("EXAMPLES that count (sentences from on-topic documents; illustrations, "
                     "not a complete list):")
NEG_EXAMPLES_HEAD = ("EXAMPLES that do NOT count (similar-looking sentences; illustrations, "
                     "not a complete list):")


@dataclass
class Component:
    """One condition a document either meets or doesn't."""

    id: str  # short formula name: W, F, R, B1 …
    condition: str  # one checkable claim about the document
    kind: str = "atom"  # atom | relation (links other components)
    links: list[str] = field(default_factory=list)  # relation: the components it connects
    yes_when: list[str] = field(default_factory=list)  # edge cases decided as met
    no_when: list[str] = field(default_factory=list)  # edge cases decided as not met

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.id):
            raise ValueError(f"component id {self.id!r}: letters, digits and _ only")
        if self.kind not in COMPONENT_KINDS:
            raise ValueError(f"component {self.id}: unknown kind {self.kind!r}")
        if self.kind == "relation" and not self.links:
            raise ValueError(f"component {self.id}: a relation must name the components it links")


# --------------------------------------------------------------------------- #
# Formulas: ids, ~ (not), & (and), | (or), parentheses. ~ binds tightest, then &, then |.
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"\s*(?:([A-Za-z][A-Za-z0-9_]*)|(.))")


def parse_formula(text: str):
    """``"P & (R | ~B1)"`` → nested tuples: ``("and", "P", ("or", "R", ("not", "B1")))``."""
    toks = []
    for name, op in _TOKEN.findall(text):
        if name:
            toks.append(name)
        elif op.strip():
            if op not in "&|~()":
                raise ValueError(f"formula {text!r}: unexpected {op!r}")
            toks.append(op)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take(expect=None):
        nonlocal pos
        t = peek()
        if t is None or (expect and t != expect):
            raise ValueError(f"formula {text!r}: expected {expect or 'a term'}, got {t!r}")
        pos += 1
        return t

    def disj():
        node = conj()
        while peek() == "|":
            take()
            node = ("or", node, conj())
        return node

    def conj():
        node = neg()
        while peek() == "&":
            take()
            node = ("and", node, neg())
        return node

    def neg():
        if peek() == "~":
            take()
            return ("not", neg())
        if peek() == "(":
            take("(")
            node = disj()
            take(")")
            return node
        t = take()
        if t in "&|~()":
            raise ValueError(f"formula {text!r}: expected a component id, got {t!r}")
        return t

    node = disj()
    if peek() is not None:
        raise ValueError(f"formula {text!r}: trailing {peek()!r}")
    return node


def formula_ids(text: str) -> set[str]:
    """Component ids a formula refers to."""
    def walk(n):
        return {n} if isinstance(n, str) else set().union(*(walk(c) for c in n[1:]))
    return walk(parse_formula(text))


def evaluate(text: str, values: dict[str, bool]) -> bool:
    """Crisp verdict from per-component yes/no."""
    def ev(n):
        if isinstance(n, str):
            return bool(values[n])
        if n[0] == "not":
            return not ev(n[1])
        a, b = ev(n[1]), ev(n[2])
        return (a and b) if n[0] == "and" else (a or b)
    return ev(parse_formula(text))


def soft_evaluate(text: str, p: dict[str, float]) -> float:
    """Probability of the formula treating components as independent: a surrogate score,
    not a calibrated probability (components of one idea are correlated)."""
    def ev(n):
        if isinstance(n, str):
            return float(p[n])
        if n[0] == "not":
            return 1.0 - ev(n[1])
        a, b = ev(n[1]), ev(n[2])
        return a * b if n[0] == "and" else 1.0 - (1.0 - a) * (1.0 - b)
    return ev(parse_formula(text))


def describe_formula(text: str) -> str:
    """``P & (R | ~B1)`` → ``[P] AND ([R] OR NOT [B1])`` — the formula for a prompt."""
    def terms(n, op):  # flatten a chain of the same operator: A & B & C, not (A & B) & C
        return terms(n[1], op) + terms(n[2], op) if not isinstance(n, str) and n[0] == op else [n]

    def ds(n, top=False):
        if isinstance(n, str):
            return f"[{n}]"
        if n[0] == "not":
            return f"NOT {ds(n[1])}"
        s = f" {n[0].upper()} ".join(ds(t) for t in terms(n, n[0]))
        return s if top else f"({s})"
    return ds(parse_formula(text), top=True)


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
    components: list[Component] = field(default_factory=list)  # structured: the conditions
    formula: str | None = None  # structured: the membership rule over component ids
    near_misses: dict[str, str] = field(default_factory=dict)  # name → formula (what's missing)

    def __post_init__(self):
        self.components = [c if isinstance(c, Component) else Component(**c) for c in self.components]
        if not self.components:
            if self.formula or self.near_misses:
                raise ValueError(f"{self.topic_id}: a formula needs components")
            return
        ids = [c.id for c in self.components]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.topic_id}: duplicate component ids")
        if not self.formula:
            raise ValueError(f"{self.topic_id}: structured codebook has no formula")
        for name, f in {"formula": self.formula, **self.near_misses}.items():
            if missing := formula_ids(f) - set(ids):
                raise ValueError(f"{self.topic_id}: {name} uses unknown components {sorted(missing)}")
        for c in self.components:
            if missing := set(c.links) - set(ids):
                raise ValueError(f"{self.topic_id}: {c.id} links unknown components {sorted(missing)}")

    @property
    def structured(self) -> bool:
        return bool(self.components)

    def component(self, cid: str) -> Component:
        return next(c for c in self.components if c.id == cid)

    def verdict(self, values: dict[str, bool]) -> bool:
        return evaluate(self.formula, values)

    def missing(self, values: dict[str, bool]) -> list[str]:
        """Near misses a document matches — which piece it lacks."""
        return [n for n, f in self.near_misses.items() if evaluate(f, values)]

    def to_dict(self) -> dict:
        d = asdict(self)
        if not self.components:  # flat codebooks save as before
            for k in ("components", "formula", "near_misses"):
                d.pop(k)
        return d

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
    # Structured codebooks say outright that examples illustrate rather than enumerate: a
    # judge reads an unmarked list as the complete specification. Flat codebooks keep their
    # wording so frozen hashes and caches still match.
    pos_head = (POS_EXAMPLES_HEAD if cb.structured
                else "POSITIVE examples (sentences from on-topic documents):")
    out = (
        f"TOPIC: {cb.label}\n"
        f"DEFINITION: {cb.definition}\n"
        f"{pos_head}\n{pos}"
    )
    # Drafted sections only when present, so template codebooks render byte-identically
    # (and keep their judge-cache entries).
    if cb.include:
        out += "\nCOUNTS when the document:\n" + "\n".join(f"  - {r}" for r in cb.include)
    if cb.exclude:
        out += "\nDOES NOT COUNT:\n" + "\n".join(f"  - {r}" for r in cb.exclude)
    if cb.structured:
        out += "\n" + render_structure(cb)
    if cb.negative_examples:
        neg_head = (NEG_EXAMPLES_HEAD if cb.structured
                    else "NEGATIVE examples (near misses — similar wording, NOT this topic):")
        out += f"\n{neg_head}\n" + "\n".join(f"  - {t}" for t in cb.negative_examples)
    return out


def render_component(c: Component) -> str:
    line = f"[{c.id}] {c.condition}"
    if c.kind == "relation":
        line += f" (links {', '.join(f'[{x}]' for x in c.links)}: they must hold of the same claim)"
    for e in c.yes_when:
        line += f"\n      {ALSO_MET}: {e}"
    for e in c.no_when:
        line += f"\n      {NOT_MET}: {e}"
    return line


def render_structure(cb: Codebook) -> str:
    """Conditions, the rule that combines them, and the named near misses."""
    lines = ["CONDITIONS (each one checked on its own):"]
    lines += [f"  {render_component(c)}" for c in cb.components]
    lines.append(f"COUNTS only when: {describe_formula(cb.formula)}")
    if cb.near_misses:
        lines.append("NEAR MISSES (do NOT count — the piece they lack):")
        lines += [f"  - {n}: {describe_formula(f)}" for n, f in cb.near_misses.items()]
    return "\n".join(lines)


def save(codebooks: list[Codebook], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.to_dict() for c in codebooks], indent=2))
    return path


def load(path: str | Path) -> list[Codebook]:
    return [Codebook(**d) for d in json.loads(Path(path).read_text())]


# --------------------------------------------------------------------------- #
# Markdown: the editable form of a codebook (round-trips through from_markdown)
# --------------------------------------------------------------------------- #
_MD_SECTIONS = ("Definition", "Rule", "Conditions", "Near misses", "Include", "Exclude",
                "Positive examples", "Negative examples", "Notes")


def to_markdown(cb: Codebook) -> str:
    """A codebook as a markdown file a person can edit; :func:`from_markdown` reads it back."""
    out = [f"# {cb.topic_id}: {cb.label}", "", "## Definition", "", cb.definition, ""]
    if cb.structured:
        out += ["## Rule", "", f"Counts only when: `{cb.formula}`", "",
                "Operators: `&` and, `|` or, `~` not, parentheses. Use the condition ids below.",
                "", "## Conditions", ""]
        for c in cb.components:
            kind = f"relation (links {', '.join(c.links)})" if c.kind == "relation" else "atom"
            out += [f"### {c.id} — {kind}", "", c.condition]
            out += [f"- {ALSO_MET}: {e}" for e in c.yes_when]
            out += [f"- {NOT_MET}: {e}" for e in c.no_when]
            out.append("")
        if cb.near_misses:
            out += ["## Near misses", ""]
            out += [f"- {n}: `{f}`" for n, f in cb.near_misses.items()]
            out.append("")
    for title, items in (("Include", cb.include), ("Exclude", cb.exclude),
                         ("Positive examples", cb.positive_examples),
                         ("Negative examples", cb.negative_examples)):
        if items:
            out += [f"## {title}", ""] + [f"- {x}" for x in items] + [""]
    if cb.notes:
        out += ["## Notes", "", cb.notes, ""]
    return "\n".join(out)


def from_markdown(text: str, *, base: Codebook | None = None) -> Codebook:
    """Read a codebook written by :func:`to_markdown` (after a person edited it). Fields the
    file doesn't carry (``source_topic_id``) come from ``base``; the result is unfrozen."""
    title = re.search(r"^# (\S+?): (.+)$", text, re.M)
    if not title:
        raise ValueError("codebook markdown needs a '# <topic id>: <label>' title")
    sections: dict[str, str] = {}
    for m in re.finditer(r"^## (.+?)\s*$\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        if m.group(1) not in _MD_SECTIONS:
            raise ValueError(f"unknown section '## {m.group(1)}' (expected one of {_MD_SECTIONS})")
        sections[m.group(1)] = m.group(2).strip()

    def bullets(name):
        return [b.strip() for b in re.findall(r"^- (.+)$", sections.get(name, ""), re.M)]

    components = []
    for m in re.finditer(r"^### (\S+) — (atom|relation)(?: \(links ([^)]*)\))?\s*$\n(.*?)(?=^### |\Z)",
                         sections.get("Conditions", ""), re.M | re.S):
        body = m.group(4).strip().splitlines()
        cond = " ".join(l.strip() for l in body if l.strip() and not l.startswith("- "))
        yes = [l[len(f"- {ALSO_MET}:"):].strip() for l in body if l.startswith(f"- {ALSO_MET}:")]
        no = [l[len(f"- {NOT_MET}:"):].strip() for l in body if l.startswith(f"- {NOT_MET}:")]
        links = [x.strip() for x in (m.group(3) or "").split(",") if x.strip()]
        components.append(Component(m.group(1), cond, m.group(2), links, yes, no))
    formula = None
    if rule := re.search(r"`([^`]+)`", sections.get("Rule", "")):
        formula = rule.group(1)
    near = {}
    for b in bullets("Near misses"):
        nm = re.fullmatch(r"(.+?):\s*`([^`]+)`", b)
        if not nm:
            raise ValueError(f"near miss {b!r}: write it as '- name: `formula`'")
        near[nm.group(1)] = nm.group(2)
    return Codebook(
        topic_id=title.group(1), label=title.group(2).strip(),
        definition=" ".join(sections.get("Definition", "").split()),
        positive_examples=bullets("Positive examples"), negative_examples=bullets("Negative examples"),
        include=bullets("Include"), exclude=bullets("Exclude"),
        source_topic_id=base.source_topic_id if base else None,
        notes=sections.get("Notes") or None,
        components=components, formula=formula, near_misses=near,
    )
