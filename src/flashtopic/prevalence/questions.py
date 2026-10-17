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

Group questions (parent/sibling) are generated once per group and shared by its members;
:func:`build_question_set` dedups them and records which qids belong to which topic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from flashtopic.prevalence.classifier import Question
from flashtopic.prevalence.codebook import Codebook
from flashtopic.prevalence.topics import Topic

QUESTION_VERSION = "q2"  # q2: plain noul wording, passing-mention rule in criteria_false
DEPTH_LEVELS = ("none", "passing mention", "a sentence or two", "a major part", "the main subject")
SUBSTANTIVE_LEVEL = 2  # index into DEPTH_LEVELS from which the judge would say "discusses"
NONE_OPTION = "none of these"
ROLES = ("presence", "depth", "parent", "sibling", "facet")


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
    return "facet" if tail.startswith("facet") else tail


def _criteria(topic: Topic, cb: Codebook | None) -> str:
    """Operational definition + positive examples — the codebook, as the model's criteria."""
    if cb is not None:
        lines = [f"Definition: {cb.definition}"]
        if cb.positive_examples:
            lines.append("Sentences from documents that DO count:")
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


def _label(topic: Topic) -> str:
    if not topic.label:
        raise ValueError(f"{topic.topic_id} has no label — run the labeling stage first.")
    return topic.label


def presence_question(topic: Topic, cb: Codebook | None = None) -> Question:
    return Question(
        qid=f"{topic.topic_id}.presence",
        kind="noul",
        text=f"Does the document discuss {_label(topic)}?",
        criteria=_criteria(topic, cb),
        criteria_false=_passing(_label(topic)),
    )


def depth_question(topic: Topic, cb: Codebook | None = None) -> Question:
    return Question(
        qid=f"{topic.topic_id}.depth",
        kind="score",
        text=f"How much of the document is devoted to {_label(topic)}?",
        options=DEPTH_LEVELS,
        criteria=_criteria(topic, cb),
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
) -> QuestionSet:
    """Presence (+ depth) per topic, parent + sibling per group, facets where given."""
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
        qids = [add(presence_question(t, cb))]
        if include_depth:
            qids.append(add(depth_question(t, cb)))
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
        by_topic[t.topic_id] = qids

    return QuestionSet(list(questions.values()), by_topic, choice_option)
