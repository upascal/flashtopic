"""Score-stratified gold-set design — where to spend the (LLM/human) labels.

The estimator is unbiased for any probability sample, but its *CI* is won or lost in
the design (see estimate.py's coverage experiment): a rare topic's false negatives
hide in the huge predicted-negative pool, so a uniform negative sample starves the
recall estimate and the interval under-covers. The fix is to:

* label most/all **predicted positives** (membership/entity-match = 1) — this nails
  precision, and labeling a stratum exhaustively zeroes its variance via the
  finite-population correction; and
* split the **predicted negatives** by surrogate ``score`` into bins and **oversample
  the top (borderline) bin**, where missed true positives concentrate.

Every sampled document carries a known inclusion probability ``pi`` and a stratum id,
so the design-based estimator can reweight and compute a valid variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flashtopic.prevalence.prototypes import TopicSurrogate


@dataclass
class SampleDesign:
    topic_id: str
    doc_ids: np.ndarray  # documents to label
    f_s: np.ndarray  # surrogate label at each sampled doc (from f_memb)
    pi_s: np.ndarray  # inclusion probability of each sampled doc
    strata: np.ndarray  # stratum id (0 = positives; 1..k = negative score-bins, hi→lo)
    stratum_info: list[dict]  # per-stratum {name, pop, n, pi}

    @property
    def n_gold(self) -> int:
        return len(self.doc_ids)


def _take(pool: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Sample ``min(n, |pool|)`` docs without replacement; return (sampled, pi)."""
    n = min(n, len(pool))
    if n == 0:
        return np.empty(0, int), 0.0
    sel = rng.choice(pool, n, replace=False)
    return sel, n / len(pool)


def design_sample(
    sur: TopicSurrogate,
    *,
    n_pos: int = 60,
    neg_bin_alloc: tuple[int, ...] = (60, 25, 15),
    seed: int = 0,
) -> SampleDesign:
    """Build a stratified gold design for one topic.

    Parameters
    ----------
    n_pos : target labels from the predicted-positive pool (capped at its size; a
        rare topic with fewer positives than this is labeled exhaustively).
    neg_bin_alloc : labels for each predicted-negative score bin, ordered
        **highest score first**. The default front-loads the borderline bin where
        false negatives live. ``len`` sets the number of negative bins.
    """
    rng = np.random.default_rng(seed)
    doc_ids, pi_s, strata = [], [], []
    info = []

    # Stratum 0 — predicted positives (estimate precision).
    pos_pool = np.where(sur.f_memb)[0]
    sel, pi = _take(pos_pool, n_pos, rng)
    doc_ids.append(sel)
    pi_s.append(np.full(len(sel), pi))
    strata.append(np.zeros(len(sel), int))
    info.append({"name": "positives", "pop": int(len(pos_pool)), "n": int(len(sel)), "pi": pi})

    # Strata 1..k — predicted negatives, split into equal-count score bins (hi→lo).
    neg_pool = np.where(~sur.f_memb)[0]
    n_bins = len(neg_bin_alloc)
    if len(neg_pool):
        order = neg_pool[np.argsort(sur.score[neg_pool])[::-1]]  # highest score first
        bins = np.array_split(order, n_bins)
        for i, (b, alloc) in enumerate(zip(bins, neg_bin_alloc), start=1):
            sel, pi = _take(b, alloc, rng)
            doc_ids.append(sel)
            pi_s.append(np.full(len(sel), pi))
            strata.append(np.full(len(sel), i, int))
            lo, hi = (float(sur.score[b].min()), float(sur.score[b].max())) if len(b) else (0, 0)
            info.append({"name": f"neg_score_bin{i}[{lo:.2f},{hi:.2f}]",
                         "pop": int(len(b)), "n": int(len(sel)), "pi": pi})

    doc_ids = np.concatenate(doc_ids)
    return SampleDesign(
        topic_id=sur.topic_id,
        doc_ids=doc_ids,
        f_s=sur.f_memb[doc_ids].astype(int),
        pi_s=np.concatenate(pi_s),
        strata=np.concatenate(strata),
        stratum_info=info,
    )
