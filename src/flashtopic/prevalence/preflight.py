"""Test items with known answers: check a codebook's question setup before freezing it.

A hand-written codebook often ships test sentences with expected answers, and running
them is how a configuration is checked before it meets the corpus: a question alone tends
to say yes to hard negatives that the codebook's rules rule out. A drafted topic codebook
carries the same material for free: its positive examples should come back yes, and its
near-miss negatives no.

This is a check on the *questions*, not an estimate: the items are single sentences
chosen by the drafter, so passing them says the codebook reached the model, not that the
model is accurate on whole documents. Calibrate the cut-off on gold docs for that
(:func:`calibrate_cutoff`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flashtopic.prevalence.classifier import DocClassifier, classify_docs, _DEFAULT_CACHE_DIR
from flashtopic.prevalence.codebook import Codebook
from flashtopic.prevalence.questions import QuestionSet


@dataclass
class CheckItem:
    text: str
    expected: bool  # True: a positive example; False: a near-miss negative

    @property
    def doc_id(self) -> int:
        """Content-derived id, so the classifier cache keys on the sentence itself and
        never collides with a corpus document id (those are small non-negative ints)."""
        return -int(hashlib.sha256(self.text.encode()).hexdigest()[:12], 16) - 1


@dataclass
class PreflightResult:
    topic_id: str
    qid: str
    items: list[CheckItem]
    p_yes: list[float | None]  # aligned with items; None when the call failed
    threshold: float = 0.5
    notes: list[str] = field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(1 for it, p in zip(self.items, self.p_yes)
                   if p is not None and (p >= self.threshold) == it.expected)

    @property
    def hard_negative_p(self) -> list[float]:
        return [p for it, p in zip(self.items, self.p_yes) if not it.expected and p is not None]

    def failures(self) -> list[tuple[CheckItem, float | None]]:
        return [(it, p) for it, p in zip(self.items, self.p_yes)
                if p is None or (p >= self.threshold) != it.expected]

    def summary(self) -> str:
        neg = ", ".join(f"{p:.2f}" for p in self.hard_negative_p) or "none"
        return (f"{self.topic_id}: {self.n_pass}/{len(self.items)} test items pass at "
                f"{self.threshold:.2f}; near-miss p = {neg}")


def check_items(cb: Codebook) -> list[CheckItem]:
    """Positive examples (expect yes) then near-miss negatives (expect no), deduplicated."""
    seen: set[str] = set()
    out = []
    for text, exp in [(t, True) for t in cb.positive_examples] + [(t, False) for t in cb.negative_examples]:
        if text and text not in seen:
            seen.add(text)
            out.append(CheckItem(text, exp))
    return out


async def preflight(
    cb: Codebook, qset: QuestionSet, clf: DocClassifier, *, threshold: float = 0.5,
    concurrency: int = 8, cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> PreflightResult:
    """Ask the topic's presence question about each test item, as if it were a document."""
    qid = f"{cb.topic_id}.presence"
    q = qset.question(qid)
    items = check_items(cb)
    res = PreflightResult(cb.topic_id, qid, items, [], threshold)
    if not items:
        res.notes.append("codebook has no examples to test")
        return res
    answers = await classify_docs(clf, {it.doc_id: it.text for it in items}, [q],
                                  concurrency=concurrency, cache_dir=cache_dir)
    for it in items:
        a = answers[it.doc_id][0]
        res.p_yes.append(a.p_yes if a.ok else None)
    if not cb.negative_examples:
        res.notes.append("no near-miss negatives: only recall is tested")
    return res


def calibrate_cutoff(
    p, y, *, grid=np.round(np.arange(0.30, 0.951, 0.01), 2), k: int = 5, seed: int = 0,
) -> tuple[float, float]:
    """Cross-validated cut-off on ``p`` that maximises agreement with gold ``y``.

    For a classify-and-count readout next to the PPI estimate. A classifier can rank well
    and still be miscalibrated on long documents, so its yes may begin well above 0.5, and
    counting at 0.5 then overstates the corpus rate. Returns
    ``(cutoff, cv_agreement)``: the median of the per-fold best cut-offs, and the mean
    held-out agreement those fold cut-offs achieve. Unweighted, so on a stratified gold
    sample it says how well ``p`` separates, not the corpus rate at that cut-off.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    ok = ~np.isnan(p)
    p, y = p[ok], y[ok]
    if len(p) < k:
        raise ValueError(f"need at least {k} labeled probabilities, got {len(p)}")
    folds = np.random.default_rng(seed).permutation(len(p)) % k
    cuts, accs = [], []
    for f in range(k):
        tr, te = folds != f, folds == f
        agree = [((p[tr] >= c) == y[tr]).mean() for c in grid]
        c = float(grid[int(np.argmax(agree))])
        cuts.append(c)
        accs.append(float(((p[te] >= c) == y[te]).mean()))
    return float(np.median(cuts)), float(np.mean(accs))
