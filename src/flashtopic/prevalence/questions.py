"""Turn a Topic into the typed questions a decision model answers about a document.

The question set is **pre-registered**: it is generated from topic structure alone (label,
codebook, hierarchy, siblings) and never from gold residuals, so the gold set is not used
twice (once to pick the questions, once to measure their error). Every question maps onto
a feature role that the surrogate regression sees under the same name for every topic:

* ``presence`` — noul: does the document discuss the topic? (codebook in ``criteria``)
* ``depth``    — score over :data:`DEPTH_LEVELS`: how much of the document is devoted to
  it? ``P(level >= "a sentence or two")`` is the direct analogue of the judge's
  "substantively discusses" — the passing-mention rule is the main source of naive bias.
* ``parent``   — noul, shared by a meta-topic's siblings: is the broader theme present?
* ``sibling``  — choice, one per meta-topic group: which sibling facet is addressed most,
  or none of these? Gives every sibling its own probability from a single question.
* ``facet``    — optional narrow nouls for a covertly mixed cluster's sub-facets.
* ``rule``     — noul per drafted include rule: a decomposed check of one part of the
  codebook, pairing the direct judgement with component checks.
* ``nearmiss`` — noul per drafted exclude rule (bar the passing-mention one): is the
  document doing the neighbouring idea instead?
* ``comp``     — noul per component of a structured codebook (``T1.comp_R``). Asked
  alongside ``presence``, never instead of it: the point-blank question (definition and
  examples only, not the structure) and the formula over components are two separate
  readings, and where they disagree is worth a human look.

Group questions (parent/sibling) are generated once per group and shared by its members;
:func:`build_question_set` dedups them and records which qids belong to which topic.

Where the codebook goes is a setting (``codebook_in``): the noul ``criteria`` slots (the
default, validated on topics), appended to the ``instructions``, or both. Which works best
varies by model and corpus: measure it on held-out gold rather than assume it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from flashtopic.prevalence.classifier import Question
from flashtopic.prevalence.codebook import Codebook, Component
from flashtopic.prevalence.topics import Topic

QUESTION_VERSION = "q4"  # q4: structured codebooks — per-component checks
DEPTH_LEVELS = ("none", "passing mention", "a sentence or two", "a major part", "the main subject")
SUBSTANTIVE_LEVEL = 2  # index into DEPTH_LEVELS from which the judge would say "discusses"
NONE_OPTION = "none of these"
ROLES = ("presence", "depth", "parent", "sibling", "facet", "rule", "nearmiss", "comp")
_NUMBERED_ROLES = ("facet", "rule", "nearmiss", "comp")  # qid tails carry an index: facet1, comp_R …
CODEBOOK_PLACEMENTS = ("criteria", "instructions", "both")


@dataclass
class TopicContext:
    """What a Topic doesn't carry itself: hierarchy + siblings (all optional)."""

    meta_theme: str | None = None  # parent label (AtomLabel.meta_theme / meta-cluster label)
    group_id: str | None = None  # id of the sibling group (e.g. meta-cluster id)
    sibling_labels: tuple[str, ...] = ()  # labels of all topics in the group, self included
    differs: str | None = None  # contrastive "how this differs from its siblings"
    facets: tuple[str, ...] = ()  # AtomLabel.discerned_subtopics for a 'mixed' cluster


def role_of(qid: str) -> str:
    """``T3.presence`` → ``presence``; ``group:2.sibling`` → ``sibling``; ``T3.facet1`` → ``facet``."""
    tail = qid.rsplit(".", 1)[-1]
    return next((r for r in _NUMBERED_ROLES if tail.startswith(r)), tail)


def _criteria(topic: Topic, cb: Codebook | None) -> str:
    """Operational definition + positive examples — the codebook, as the model's criteria."""
    if cb is not None:
        lines = [f"Definition: {cb.definition}"]
        if cb.include:
            lines.append("It counts when the document:")
            lines += [f"- {r}" for r in cb.include]
        if cb.positive_examples:
            lines.append("Examples of sentences that count (illustrations, not a complete list):"
                         if cb.structured else "Sentences from documents that DO count:")
            lines += [f"- {e}" for e in cb.positive_examples[:5]]
        return "\n".join(lines)
    terms = ", ".join(topic.distinctive_terms[:6])
    lines = [f"Topic: {topic.label}" + (f" (themes/terms: {terms})" if terms else "")]
    if topic.definition:
        lines.append(f"Broader theme: {topic.definition}")
    if topic.exemplars:
        lines.append("Sentences from documents that DO count:")
        lines += [f"- {e}" for e in topic.exemplars[:5]]
    return "\n".join(lines)


def _passing(subject: str) -> str:
    """The "no" side of a noul: absent, or only a passing mention.

    The passing-mention rule lives here rather than in the question text: a live A/B on
    jev-1.13 (2026-09-28) found "substantively discuss X, beyond a passing mention?" drove
    clear positives to ~0.1, while "discuss X?" + this criterion kept positives > 0.9 and
    still pushed a one-clause date reference down to ~0.1.
    """
    return (f"{subject} is absent, or mentioned only in passing (a name-drop or date "
            "reference) while the document is about something else.")


def _criteria_false(subject: str, cb: Codebook | None) -> str:
    """The "no" side: the passing-mention rule, then the drafted exclude rules and near misses.

    Without these rules a classifier tends to say yes to hard negatives that merely touch the
    topic; with them it can tell the neighbouring idea apart.
    """
    lines = [_passing(subject)]
    if cb is not None and cb.exclude:
        lines.append("It does not count when the document:")
        lines += [f"- {r}" for r in cb.exclude]
    if cb is not None and cb.negative_examples:
        lines.append("Examples that look similar but do NOT count (illustrations, not a "
                     "complete list):" if cb.structured
                     else "Sentences that look similar but do NOT count:")
        # Drafts carry at most 4; hand-edited codebooks may add near misses up to 6.
        lines += [f"- {e}" for e in cb.negative_examples[:6]]
    return "\n".join(lines)


def _codebook_block(topic: Topic, cb: Codebook | None) -> str:
    return (f"Codebook.\nCounts as yes:\n{_criteria(topic, cb)}\n\n"
            f"Counts as no:\n{_criteria_false(_label(topic), cb)}")


def _check_placement(codebook_in: str) -> None:
    if codebook_in not in CODEBOOK_PLACEMENTS:
        raise ValueError(f"codebook_in must be one of {CODEBOOK_PLACEMENTS}, got {codebook_in!r}")


def _label(topic: Topic) -> str:
    if not topic.label:
        raise ValueError(f"{topic.topic_id} has no label — run the labeling stage first.")
    return topic.label


def presence_question(topic: Topic, cb: Codebook | None = None, *,
                      codebook_in: str = "criteria") -> Question:
    _check_placement(codebook_in)
    text = f"Does the document discuss {_label(topic)}?"
    if codebook_in == "criteria":
        true, false = _criteria(topic, cb), _criteria_false(_label(topic), cb)
    else:
        text = f"{text}\n\n{_codebook_block(topic, cb)}"
        true, false = ((_criteria(topic, cb), _criteria_false(_label(topic), cb))
                       if codebook_in == "both"
                       else ("The document meets the codebook above.", _passing(_label(topic))))
    return Question(qid=f"{topic.topic_id}.presence", kind="noul", text=text,
                    criteria=true, criteria_false=false)


def depth_question(topic: Topic, cb: Codebook | None = None, *,
                   codebook_in: str = "criteria") -> Question:
    # Score questions have no criteria slot: to_wire appends criteria to the instructions,
    # so the placements differ only in how much of the codebook is shown.
    _check_placement(codebook_in)
    return Question(
        qid=f"{topic.topic_id}.depth",
        kind="score",
        text=f"How much of the document is devoted to {_label(topic)}?",
        options=DEPTH_LEVELS,
        criteria=_criteria(topic, cb) if codebook_in == "criteria" else _codebook_block(topic, cb),
    )


def parent_question(group_id: str, meta_theme: str, sibling_labels: tuple[str, ...] = ()) -> Question:
    facets = ", ".join(dict.fromkeys(l for l in sibling_labels if l))
    true = f"The document discusses {meta_theme}" + (
        f", including any of its facets such as: {facets}." if facets else ".")
    return Question(
        qid=f"group:{group_id}.parent",
        kind="noul",
        text=f"Does the document discuss {meta_theme}?",
        criteria=true,
        criteria_false=_passing(meta_theme),
    )


def sibling_question(group_id: str, labels: tuple[str, ...], meta_theme: str | None = None) -> Question:
    opts = tuple(dict.fromkeys(labels))  # dedup, keep order
    if len(opts) > 254:
        raise ValueError(f"group {group_id}: {len(opts)} siblings exceed the 255-option cap")
    if len(opts) < 1:
        raise ValueError(f"group {group_id}: no sibling labels")
    about = f" of {meta_theme}" if meta_theme else ""
    return Question(
        qid=f"group:{group_id}.sibling",
        kind="choice",
        text=(f"Which of these facets{about} does the document address most substantively? "
              f"Answer '{NONE_OPTION}' if it addresses none of them beyond a passing mention."),
        options=opts + (NONE_OPTION,),
    )


def facet_questions(topic: Topic, facets: tuple[str, ...]) -> list[Question]:
    return [
        Question(
            qid=f"{topic.topic_id}.facet{i}",
            kind="noul",
            text=f"Does the document discuss {f}?",
            criteria=f"The document discusses {f}.",
            criteria_false=_passing(f),
        )
        for i, f in enumerate(facets, 1)
    ]


def _is_passing_rule(rule: str) -> bool:
    return "passing" in rule.lower()


def _condition_question(qid: str, rule: str) -> Question:
    return Question(
        qid=qid,
        kind="noul",
        text=f"Does this describe the document? {rule}",
        criteria=f"The document does this: {rule}",
        criteria_false="The document does not do this, or touches on it only in passing.",
    )


def rule_questions(topic: Topic, cb: Codebook) -> list[Question]:
    """One noul per include rule: a decomposed check, like SL_NEG / SL_POS / SL_LINK."""
    return [_condition_question(f"{topic.topic_id}.rule{i}", r) for i, r in enumerate(cb.include, 1)]


def nearmiss_questions(topic: Topic, cb: Codebook) -> list[Question]:
    """One noul per exclude rule. The passing-mention rule is skipped: depth measures it."""
    rules = [r for r in cb.exclude if not _is_passing_rule(r)]
    return [_condition_question(f"{topic.topic_id}.nearmiss{i}", r) for i, r in enumerate(rules, 1)]


def component_question(topic: Topic, cb: Codebook, c: Component) -> Question:
    """One condition of a structured codebook, with its decided edge cases.

    A relation names the conditions it links, so Jev judges the connection rather than
    the co-occurrence: "W and F both appear" is not "F is how W was overcome".
    """
    text = f"Does this describe the document? {c.condition}"
    if c.kind == "relation":
        parts = "\n".join(f"- [{x}] {cb.component(x).condition}" for x in c.links)
        text += ("\n\nThis is about how these conditions connect, not whether each appears "
                 f"somewhere:\n{parts}")
    true = [f"The document does this: {c.condition}"] + [
        f"- Also counts (in addition to the above): {e}" for e in c.yes_when]
    false = ["The document does not do this."] + [f"- Does not count: {e}" for e in c.no_when]
    if c.kind == "relation":
        false.append("- Does not count: the linked conditions each appear, but in separate "
                     "claims or without this connection.")
    return Question(qid=f"{topic.topic_id}.comp_{c.id}", kind="noul", text=text,
                    criteria="\n".join(true), criteria_false="\n".join(false))


def component_questions(topic: Topic, cb: Codebook) -> list[Question]:
    return [component_question(topic, cb, c) for c in cb.components]


@dataclass
class QuestionSet:
    questions: list[Question]
    by_topic: dict[str, list[str]]  # topic_id → qids (own + shared group questions)
    choice_option: dict[str, str] = field(default_factory=dict)  # topic_id → its sibling option
    hash: str = ""

    def question(self, qid: str) -> Question:
        return self._index[qid]

    def __post_init__(self):
        self._index = {q.qid: q for q in self.questions}
        if not self.hash:
            self.hash = question_set_hash(self.questions)

    def for_topic(self, topic_id: str) -> list[Question]:
        return [self._index[q] for q in self.by_topic.get(topic_id, [])]


def question_set_hash(questions) -> str:
    """Order-independent hash of the question contents (+ version)."""
    h = hashlib.sha256(QUESTION_VERSION.encode())
    for c in sorted(q.content_hash for q in questions):
        h.update(c.encode())
    return h.hexdigest()[:16]


def build_question_set(
    topics: list[Topic],
    ctx_by_topic: dict[str, TopicContext] | None = None,
    codebooks: dict[str, Codebook] | None = None,
    *,
    include_depth: bool = True,
    include_facets: bool = True,
    include_rules: bool = True,
    codebook_in: str = "criteria",
) -> QuestionSet:
    """Presence (+ depth) per topic, parent + sibling per group, facets where given,
    rule / near-miss checks where a drafted codebook has include / exclude rules, and
    component checks (next to presence) where it is structured."""
    _check_placement(codebook_in)
    ctx_by_topic = ctx_by_topic or {}
    codebooks = codebooks or {}
    questions: dict[str, Question] = {}
    by_topic: dict[str, list[str]] = {}
    choice_option: dict[str, str] = {}

    def add(q: Question) -> str:
        questions.setdefault(q.qid, q)
        return q.qid

    for t in topics:
        cb = codebooks.get(t.topic_id)
        ctx = ctx_by_topic.get(t.topic_id, TopicContext())
        qids = [add(presence_question(t, cb, codebook_in=codebook_in))]
        if include_depth:
            qids.append(add(depth_question(t, cb, codebook_in=codebook_in)))
        if ctx.group_id is not None:
            if ctx.meta_theme:
                qids.append(add(parent_question(ctx.group_id, ctx.meta_theme, ctx.sibling_labels)))
            labels = ctx.sibling_labels or (t.label,)
            if t.label not in labels:
                labels = labels + (t.label,)
            if len(labels) >= 2:
                qids.append(add(sibling_question(ctx.group_id, labels, ctx.meta_theme)))
                choice_option[t.topic_id] = t.label
        if include_facets and ctx.facets:
            qids += [add(q) for q in facet_questions(t, ctx.facets)]
        if include_rules and cb is not None:
            qids += [add(q) for q in rule_questions(t, cb) + nearmiss_questions(t, cb)]
        if cb is not None and cb.structured:
            qids += [add(q) for q in component_questions(t, cb)]
        by_topic[t.topic_id] = qids

    return QuestionSet(list(questions.values()), by_topic, choice_option)
