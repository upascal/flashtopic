"""Cross-fitted surrogate models: a calibrated per-document ``f`` from features + gold.

The PPI estimator is unbiased for **any** surrogate; the surrogate only sets the CI width,
through the within-stratum variance of ``f - Y`` (``estimate.ppi_analytic_ci``). So the
job here is to make ``f`` predict the gold label as well as possible — and to do it
without letting a document's own label leak into its own prediction, which would make the
measured residuals (and the CI) optimistic.

**Cross-prediction** (Zrnic & Candès): K folds by document; the model fit on the other
folds predicts fold k's gold documents *and* the whole corpus. ``f_gold`` is out-of-fold;
``f_pop`` is the mean of the K fold models. The same ``f`` is then used on both sides of
the estimator, as PPI requires. Folds are assigned by a hash of ``doc_idx`` so a document
sampled for several topics never influences its own prediction for any of them.

Three candidates, chosen by the design SE they produce (a fixed rule, so the selection
itself barely touches the gold):

* ``memb``           — today's membership indicator (the baseline to beat)
* ``presence_platt`` — the classifier's raw presence probability, 2-parameter recalibrated
  (safe when there are only a handful of gold positives)
* ``lr``             — L2 logistic regression on all features (needs ``MIN_POS_FOR_LR``)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from flashtopic.prevalence.estimate import ppi_analytic_ci

MIN_POS_FOR_LR = 10
K_FOLDS = 5
MIN_ESS = 40.0  # Kish effective sample size below which 1/pi weights are tempered
_LOGIT_CLIP = 6.0


@dataclass
class FittedSurrogate:
    topic_id: str
    kind: str  # memb | presence_platt | lr | det_lr
    f_gold: np.ndarray  # (n_gold,) out-of-fold prediction per gold doc
    se_gold: float  # design SE of the PPI rectifier with these residuals
    f_pop: np.ndarray | None = None  # (N,) mean of fold models; None when X_pop not given
    feature_names: list[str] = field(default_factory=list)
    coef_mean: np.ndarray | None = None  # standardized coefficients, mean over folds
    intercept_mean: float = float("nan")
    n_folds: int = 0
    ess: float = float("nan")
    weight_power: float = 1.0
    n_pos: int = 0

    def coef_table(self) -> list[tuple[str, float]]:
        if self.coef_mean is None:
            return []
        order = np.argsort(-np.abs(self.coef_mean))
        return [(self.feature_names[i], float(self.coef_mean[i])) for i in order]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def doc_folds(doc_ids, k: int = K_FOLDS, seed: int = 0) -> np.ndarray:
    """Stable fold id per document (hash of doc_idx + seed) — identical across topics."""
    out = np.empty(len(doc_ids), dtype=int)
    for i, d in enumerate(doc_ids):
        h = hashlib.sha256(f"{seed}:{int(d)}".encode()).digest()
        out[i] = int.from_bytes(h[:4], "big") % k
    return out


def kish_ess(w) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.maximum((w**2).sum(), 1e-12))


def design_se(f_gold, y, pi, strata, n_docs: int) -> float:
    """The analytic PPI standard error implied by these gold residuals (point irrelevant)."""
    return ppi_analytic_ci(0.0, f_gold, y, pi, strata, n_docs)[2]


def logit(p, clip: float = _LOGIT_CLIP) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
def baseline_memb(topic_id, f_memb_gold, y, pi, strata, n_docs, f_memb_pop=None) -> FittedSurrogate:
    f = np.asarray(f_memb_gold, float)
    return FittedSurrogate(
        topic_id=topic_id, kind="memb", f_gold=f,
        se_gold=design_se(f, y, pi, strata, n_docs),
        f_pop=None if f_memb_pop is None else np.asarray(f_memb_pop, float),
        n_pos=int(np.sum(y)),
    )


def fit_cross_fitted(
    topic_id: str,
    X_gold: np.ndarray,
    y: np.ndarray,
    pi: np.ndarray,
    strata: np.ndarray,
    doc_ids_gold: np.ndarray,
    X_pop: np.ndarray | None = None,
    *,
    n_docs: int,
    feature_names: list[str] | None = None,
    k: int = K_FOLDS,
    C: float = 1.0,
    weight_power: float = 1.0,
    seed: int = 0,
    kind: str = "lr",
) -> FittedSurrogate:
    """Cross-fitted weighted L2 logistic regression → out-of-fold ``f_gold`` (+ ``f_pop``).

    Sample weights are ``(1/pi) ** weight_power``: 1/pi matches the design variance the
    estimator actually pays for, but spans orders of magnitude across strata, so when the
    Kish ESS drops below ``MIN_ESS`` the power is tempered to 0.5.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_gold = np.asarray(X_gold, float)
    y = np.asarray(y, int)
    pi = np.asarray(pi, float)
    n, p = X_gold.shape
    names = list(feature_names) if feature_names else [f"x{i}" for i in range(p)]

    w = (1.0 / pi) ** weight_power
    ess = kish_ess(w)
    if ess < MIN_ESS and weight_power > 0.5:
        weight_power = 0.5
        w = (1.0 / pi) ** weight_power
        ess = kish_ess(w)

    folds = doc_folds(doc_ids_gold, k, seed)
    f_gold = np.full(n, np.nan)
    f_pop_acc = np.zeros(X_pop.shape[0]) if X_pop is not None else None
    coefs, intercepts, used = [], [], 0
    for fk in range(k):
        test = folds == fk
        train = ~test
        if train.sum() == 0:
            continue
        y_tr = y[train]
        if y_tr.min() == y_tr.max():  # one class in the training folds → weighted constant
            const = float(np.average(y_tr, weights=w[train]))
            const = min(max(const, 1e-3), 1 - 1e-3)
            f_gold[test] = const
            if f_pop_acc is not None:
                f_pop_acc += const
            used += 1
            continue
        scaler = StandardScaler().fit(X_gold[train], sample_weight=w[train])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(scaler.transform(X_gold[train]), y_tr, sample_weight=w[train])
        if test.any():
            f_gold[test] = clf.predict_proba(scaler.transform(X_gold[test]))[:, 1]
        if f_pop_acc is not None:
            f_pop_acc += clf.predict_proba(scaler.transform(X_pop))[:, 1]
        coefs.append(clf.coef_[0])
        intercepts.append(float(clf.intercept_[0]))
        used += 1

    # A fold with no test rows leaves NaN only where no gold doc lives — none by construction;
    # guard anyway for tiny gold sets where a fold may be empty in *train* (skipped above).
    if np.isnan(f_gold).any():
        f_gold[np.isnan(f_gold)] = float(np.average(y, weights=w))

    return FittedSurrogate(
        topic_id=topic_id, kind=kind, f_gold=f_gold,
        se_gold=design_se(f_gold, y, pi, strata, n_docs),
        f_pop=None if f_pop_acc is None else f_pop_acc / max(used, 1),
        feature_names=names,
        coef_mean=np.mean(coefs, axis=0) if coefs else None,
        intercept_mean=float(np.mean(intercepts)) if intercepts else float("nan"),
        n_folds=used, ess=ess, weight_power=weight_power, n_pos=int(y.sum()),
    )


def platt_presence(topic_id, p_gold, y, pi, strata, doc_ids_gold, p_pop=None, *, n_docs, **kw) -> FittedSurrogate:
    """Raw presence probability, recalibrated with 2 parameters (σ(a·logit p + b)), cross-fitted."""
    Xg = logit(p_gold)[:, None]
    Xp = None if p_pop is None else logit(p_pop)[:, None]
    return fit_cross_fitted(topic_id, Xg, y, pi, strata, doc_ids_gold, Xp, n_docs=n_docs,
                            feature_names=["presence.logit"], kind="presence_platt", **kw)


def choose_surrogate(candidates: list[FittedSurrogate], *, min_pos: int = MIN_POS_FOR_LR) -> FittedSurrogate:
    """Smallest design SE, with ``lr``-type candidates gated on enough gold positives."""
    eligible = [c for c in candidates if not (c.kind.endswith("lr") and c.n_pos < min_pos)]
    if not eligible:
        eligible = [c for c in candidates if c.kind == "memb"] or candidates
    return min(eligible, key=lambda c: (c.se_gold if np.isfinite(c.se_gold) else np.inf))
