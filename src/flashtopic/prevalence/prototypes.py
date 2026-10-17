"""Per-document surrogate signals for each topic: the cheap whole-corpus side.

For every topic we materialize two document-level signals over **all** N_doc
documents:

* ``f_memb`` (binary) — does the document contain ≥1 of the topic's member
  sentences? This is the *current discovery method's* prediction: atom membership
  for thematic topics, the named-entity match for person topics. Its corpus mean is
  the **naive prevalence** — the (biased) number the gold set will debias.

* ``score`` (continuous) — max cosine of the document's sentences to the topic
  prototype (mean of member embeddings, L2-normalized; identical to the dedup
  driver's centroid). Defined for *every* document, member or not — so a document
  the membership predictor missed but that is semantically on-topic still scores
  high. That is what lets the sampler hunt false negatives in the huge
  predicted-negative pool, where a rare topic's CI is won or lost.

``f_memb`` is the surrogate the estimator debiases; ``score`` drives stratified
sampling. Both come straight from the cached embeddings — no LLM, no new cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flashtopic.prevalence.data import RunContext
from flashtopic.prevalence.topics import Topic


@dataclass
class TopicSurrogate:
    topic_id: str
    prototype: np.ndarray  # (d,) L2-normalized topic centroid
    f_memb: np.ndarray  # (N_doc,) bool — membership/entity-match predictor
    score: np.ndarray  # (N_doc,) float — max cosine of a doc's sentences to prototype
    naive: float  # mean(f_memb) — uncorrected prevalence

    @property
    def n_member_docs(self) -> int:
        return int(self.f_memb.sum())


def _topic_prototype(ctx: RunContext, member_uids: list[int]) -> np.ndarray:
    c = ctx.E[member_uids].mean(0)
    return c / max(np.linalg.norm(c), 1e-9)


def build_surrogate(ctx: RunContext, topic: Topic) -> TopicSurrogate:
    """Compute the membership predictor + cosine score over all documents."""
    n = ctx.n_docs
    proto = _topic_prototype(ctx, topic.member_uids)

    # Continuous score: per-uid cosine, then max over each document's sentences.
    uid_cos = ctx.E @ proto  # (U,)
    score = np.full(n, -1.0, dtype=float)
    for d, uids in ctx.doc_uids.items():
        score[d] = float(uid_cos[uids].max())

    # Binary membership predictor (== the naive baseline's indicator).
    f_memb = np.zeros(n, dtype=bool)
    member_docs = ctx.docs_of_uids(topic.member_uids)
    if member_docs:
        f_memb[list(member_docs)] = True

    return TopicSurrogate(
        topic_id=topic.topic_id,
        prototype=proto,
        f_memb=f_memb,
        score=score,
        naive=float(f_memb.mean()),
    )


def candidate_false_negatives(sur: TopicSurrogate, k: int = 10) -> np.ndarray:
    """Doc ids the membership predictor flags *negative* but that score highest —
    the documents most likely to be missed true positives. A quick gut-check that
    the surrogate has recall headroom worth measuring."""
    neg = np.where(~sur.f_memb)[0]
    return neg[np.argsort(sur.score[neg])[::-1][:k]]
