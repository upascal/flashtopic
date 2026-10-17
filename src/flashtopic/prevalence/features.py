"""Per-document feature assembly for the surrogate regression.

Two blocks, concatenated:

* **deterministic** — zero-cost signals already implied by the clustering: cosine
  ``score`` and ``f_memb`` from the :class:`TopicSurrogate`, the number of the topic's
  member sentences in the document, log document length, and how many of the topic's
  distinctive terms the document contains. All computable from the ``RunContext``'s unique
  sentences, so no full-text fetch is needed.
* **classifier** — the decision model's answers to the topic's questions, mapped to
  role-named columns (``presence.logit``, ``depth.p_substantive``, ``sibling.p_self`` …)
  so the regression coefficients mean the same thing for every topic.

Probabilities enter as clipped logits (a linear model wants them on the log-odds scale).
A missing/failed answer contributes 0 and increments the ``missing`` column, so a partial
outage degrades the surrogate instead of dropping rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flashtopic.prevalence.classifier import Answer
from flashtopic.prevalence.data import RunContext
from flashtopic.prevalence.prototypes import TopicSurrogate
from flashtopic.prevalence.questions import NONE_OPTION, SUBSTANTIVE_LEVEL, QuestionSet, role_of
from flashtopic.prevalence.surrogate import logit
from flashtopic.prevalence.topics import Topic

FEATURE_VERSION = "f1"
DET_NAMES = ["score", "f_memb", "n_member_sents", "log_doc_chars", "term_hits"]


@dataclass
class FeatureMatrix:
    doc_ids: np.ndarray
    X: np.ndarray  # (n_docs, n_features)
    names: list[str]
    version: str = FEATURE_VERSION

    def select(self, names: list[str]) -> "FeatureMatrix":
        idx = [self.names.index(n) for n in names]
        return FeatureMatrix(self.doc_ids, self.X[:, idx], list(names), self.version)

    def hstack(self, other: "FeatureMatrix") -> "FeatureMatrix":
        assert np.array_equal(self.doc_ids, other.doc_ids)
        return FeatureMatrix(self.doc_ids, np.hstack([self.X, other.X]),
                             self.names + other.names, self.version)


def _doc_text(ctx: RunContext, d: int) -> str:
    t = ctx.doc_texts.get(d)
    if t is not None:
        return t
    return "\n".join(ctx.texts[u] for u in ctx.doc_uids.get(d, []))


def deterministic_features(ctx: RunContext, topic: Topic, sur: TopicSurrogate, doc_ids) -> FeatureMatrix:
    doc_ids = np.asarray(doc_ids, int)
    members = set(int(u) for u in topic.member_uids)
    terms = [t.lower() for t in topic.distinctive_terms[:10]]
    X = np.zeros((len(doc_ids), len(DET_NAMES)))
    for i, d in enumerate(doc_ids):
        d = int(d)
        uids = ctx.doc_uids.get(d, [])
        text = _doc_text(ctx, d)
        low = text.lower()
        X[i, 0] = float(sur.score[d])
        X[i, 1] = float(sur.f_memb[d])
        X[i, 2] = float(sum(1 for u in uids if u in members))
        X[i, 3] = float(np.log1p(len(text)))
        X[i, 4] = float(sum(1 for t in terms if t and t in low))
    return FeatureMatrix(doc_ids, X, list(DET_NAMES))


def classifier_feature_names(qset: QuestionSet, topic_id: str) -> list[str]:
    names: list[str] = []
    for q in qset.for_topic(topic_id):
        role = role_of(q.qid)
        tail = q.qid.rsplit(".", 1)[-1]
        if q.kind == "noul":
            names.append(f"{tail}.logit")  # presence.logit, parent.logit, facet1.logit …
        elif q.kind == "score":
            names += [f"{role}.expect", f"{role}.p_substantive"]
        else:  # choice
            names += [f"{role}.p_self", f"{role}.p_none"]
    names.append("missing")
    return names


def classifier_features(
    answers_by_doc: dict[int, list[Answer]],
    qset: QuestionSet,
    topic: Topic,
    doc_ids,
) -> FeatureMatrix:
    doc_ids = np.asarray(doc_ids, int)
    questions = qset.for_topic(topic.topic_id)
    names = classifier_feature_names(qset, topic.topic_id)
    X = np.zeros((len(doc_ids), len(names)))
    self_opt = qset.choice_option.get(topic.topic_id)

    for i, d in enumerate(doc_ids):
        by_qid = {a.qid: a for a in answers_by_doc.get(int(d), [])}
        col = 0
        missing = 0
        for q in questions:
            a = by_qid.get(q.qid)
            ok = a is not None and a.ok
            if q.kind == "noul":
                if ok:
                    X[i, col] = logit(a.p_yes)
                else:
                    missing += 1
                col += 1
            elif q.kind == "score":
                if ok:
                    p = np.asarray(a.dist, float)
                    n = len(p)
                    X[i, col] = float((p * np.arange(n)).sum() / max(n - 1, 1))
                    X[i, col + 1] = float(logit(p[SUBSTANTIVE_LEVEL:].sum()))
                else:
                    missing += 1
                col += 2
            else:  # choice
                if ok and self_opt is not None and self_opt in q.options:
                    p = np.asarray(a.dist, float)
                    X[i, col] = float(logit(p[q.options.index(self_opt)]))
                    X[i, col + 1] = float(logit(p[q.options.index(NONE_OPTION)]))
                else:
                    missing += 1
                col += 2
        X[i, col] = float(missing)
    return FeatureMatrix(doc_ids, X, names)


def presence_probability(answers_by_doc: dict[int, list[Answer]], topic_id: str, doc_ids) -> np.ndarray:
    """The raw ``presence`` answer per doc (NaN when missing) — the zero-shot Jev score."""
    qid = f"{topic_id}.presence"
    out = np.full(len(doc_ids), np.nan)
    for i, d in enumerate(doc_ids):
        for a in answers_by_doc.get(int(d), []):
            if a.qid == qid and a.ok:
                out[i] = a.p_yes
    return out


def assemble(
    ctx: RunContext,
    topic: Topic,
    sur: TopicSurrogate,
    doc_ids,
    answers_by_doc: dict[int, list[Answer]] | None = None,
    qset: QuestionSet | None = None,
) -> FeatureMatrix:
    """Deterministic block, plus the classifier block when answers + question set are given."""
    F = deterministic_features(ctx, topic, sur, doc_ids)
    if answers_by_doc is not None and qset is not None:
        F = F.hstack(classifier_features(answers_by_doc, qset, topic, doc_ids))
    return F
