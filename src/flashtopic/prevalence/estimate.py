"""Debiased document-level prevalence estimators (ACC + PPI) with bootstrap CIs.

The whole point: we already hold the entire corpus, so the *raw* prevalence of a
topic — the fraction of documents the cheap surrogate ``f`` flags — is known
exactly. What is uncertain is whether ``f`` is *correct*. These estimators use a
small, probability-sampled, human/LLM-labeled gold set to measure ``f``'s
misclassification and subtract it, yielding an (approximately) unbiased prevalence
plus a valid confidence interval — even when ``f`` is biased.

Two design-based estimators, both unbiased regardless of how good ``f`` is:

* **ACC** (Forman's "Adjusted Classify-and-Count"): invert the confusion matrix.
  ``pi = (p_hat - FPR) / (TPR - FPR)`` where ``p_hat`` is the exact surrogate
  positive-rate over the *whole* corpus and ``TPR``/``FPR`` come from the gold set.
  Transparent; can be unstable when the surrogate is near-random (``TPR ≈ FPR``).

* **PPI** (prediction-powered inference, design-based form): correct the exact
  surrogate mean by the gold-estimated bias.
  ``pi = mean_pop(f) - HT_mean(f - Y)``. More stable; the natural primary CI.

Everything is **document-level**: one row per document, ``f_d, Y_d ∈ {0,1}``. This
is the estimand that answers "what fraction of documents discuss topic t" and it
sidesteps the sentence-within-document correlation problem (each document is one
independent unit). Stratified sampling is supported via per-document inclusion
probabilities ``pi_d`` (Horvitz–Thompson weights ``w_d = 1/pi_d``); a simple random
sample is the special case ``pi_d = n/N``.

Pure numpy — no new dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Floor for the ACC denominator |TPR - FPR|. Below this the surrogate carries
# almost no signal and ACC is ill-conditioned; we clamp and let the wide
# bootstrap CI tell the story rather than emitting an exploded point estimate.
_DENOM_FLOOR = 1e-3


@dataclass
class PrevalenceResult:
    topic_id: object
    n_docs: int  # population size N (denominator of the prevalence)
    n_gold: int  # labeled gold documents
    n_gold_pos: int  # gold documents with true label Y=1
    naive: float  # mean_pop(f) — uncorrected surrogate positive-rate
    tpr: float  # P(f=1 | Y=1), HT-weighted over the gold set
    fpr: float  # P(f=1 | Y=0), HT-weighted over the gold set
    pi_acc: float  # ACC point estimate
    pi_ppi: float  # PPI point estimate
    ci_low: float  # 95% CI lower
    ci_high: float  # 95% CI upper
    se: float  # standard error of the headline (PPI) estimate; nan if bootstrap-only
    ci_method: str  # 'ppi_analytic' (default), 'ppi_boot', or 'acc_boot'
    min_cell: int  # smallest of the 4 confusion cells in the gold set
    reliable: bool  # min_cell >= MIN_CELL — else the CI is approximate (sparse cell)

    def to_dict(self) -> dict:
        return asdict(self)


# A confusion cell with fewer than this many gold documents makes the
# corresponding rate (and therefore the CI) unreliable. For a rare topic the
# binding cell is false-negatives (f=0, Y=1) hidden in the huge predicted-negative
# pool — surface it rather than report a falsely tight interval.
MIN_CELL = 5


# --------------------------------------------------------------------------- #
# Core rate + point estimators
# --------------------------------------------------------------------------- #
def weighted_confusion_rates(
    f_s: np.ndarray, y_s: np.ndarray, w_s: np.ndarray
) -> tuple[float, float]:
    """Horvitz–Thompson estimates of ``TPR = P(f=1|Y=1)`` and ``FPR = P(f=1|Y=0)``.

    ``f_s``, ``y_s`` are gold-set surrogate and true labels (0/1); ``w_s = 1/pi_d``
    are inclusion-probability weights that undo any stratified oversampling.
    Returns ``(nan, nan)`` for a rate whose conditioning class is empty in the gold
    set (e.g. no true positives sampled → TPR undefined).
    """
    f_s = np.asarray(f_s, float)
    y_s = np.asarray(y_s, float)
    w_s = np.asarray(w_s, float)

    pos_mass = float((w_s * y_s).sum())
    neg_mass = float((w_s * (1.0 - y_s)).sum())
    tpr = float((w_s * f_s * y_s).sum() / pos_mass) if pos_mass > 0 else np.nan
    fpr = float((w_s * f_s * (1.0 - y_s)).sum() / neg_mass) if neg_mass > 0 else np.nan
    return tpr, fpr


def acc_point(p_hat: float, f_s: np.ndarray, y_s: np.ndarray, w_s: np.ndarray) -> float:
    """ACC point estimate ``(p_hat - FPR) / (TPR - FPR)`` clamped to [0, 1].

    ``p_hat`` is the exact surrogate positive-rate over the *whole* corpus.
    """
    tpr, fpr = weighted_confusion_rates(f_s, y_s, w_s)
    if np.isnan(tpr) or np.isnan(fpr):
        return np.nan
    denom = tpr - fpr
    if abs(denom) < _DENOM_FLOOR:
        denom = _DENOM_FLOOR if denom >= 0 else -_DENOM_FLOOR
    return float(np.clip((p_hat - fpr) / denom, 0.0, 1.0))


def ppi_point(
    p_hat: float, f_s: np.ndarray, y_s: np.ndarray, pi_s: np.ndarray, n_docs: int
) -> float:
    """Design-based PPI point estimate ``mean_pop(f) - HT_mean(f - Y)`` in [0, 1].

    The rectifier ``HT_mean(f - Y) = (1/N) * sum_gold (f_d - Y_d)/pi_d`` is the
    Horvitz–Thompson estimate of the population mean surrogate error; subtracting it
    debiases the exact surrogate mean ``p_hat``. Unbiased for any surrogate ``f``.
    """
    f_s = np.asarray(f_s, float)
    y_s = np.asarray(y_s, float)
    pi_s = np.asarray(pi_s, float)
    rectifier = float(((f_s - y_s) / pi_s).sum() / n_docs)
    return float(np.clip(p_hat - rectifier, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Analytic design-based PPI variance (primary CI)
# --------------------------------------------------------------------------- #
def ppi_analytic_ci(
    pi_ppi: float,
    f_s: np.ndarray,
    y_s: np.ndarray,
    pi_s: np.ndarray,
    strata: np.ndarray,
    n_docs: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Stratified design-based normal CI for the PPI estimate; returns (lo, hi, se).

    The exact surrogate mean ``p_hat`` contributes no variance (computed over the
    whole corpus), so all uncertainty is in the rectifier
    ``R = (1/N) * sum_h HT_total_h(f - Y)``. Strata are sampled independently, so
    ``Var(R) = (1/N^2) * sum_h Var(HT_total_h)`` with the Horvitz–Thompson
    stratified-SRS variance ``N_h^2 (1 - n_h/N_h) s_h^2 / n_h``. The finite-population
    correction ``(1 - n_h/N_h)`` drives a stratum's variance to zero when it is
    labeled exhaustively — i.e. labeling *all* flagged positives for a rare topic
    genuinely tightens the interval.
    """
    g = np.asarray(f_s, float) - np.asarray(y_s, float)
    pi_s = np.asarray(pi_s, float)
    strata = np.asarray(strata)

    var_r = 0.0
    for s in np.unique(strata):
        m = strata == s
        n_h = int(m.sum())
        if n_h < 2:
            continue  # single draw → no within-stratum variance estimate
        pi_h = float(pi_s[m][0])  # constant inclusion prob within a stratum
        n_big_h = n_h / pi_h  # recovered population size of the stratum
        fpc = max(0.0, 1.0 - n_h / n_big_h)
        s2 = float(np.var(g[m], ddof=1))
        var_r += n_big_h**2 * fpc * s2 / n_h
    se = float(np.sqrt(var_r) / n_docs)

    z = _normal_quantile(1 - alpha / 2)
    return (
        float(np.clip(pi_ppi - z * se, 0.0, 1.0)),
        float(np.clip(pi_ppi + z * se, 0.0, 1.0)),
        se,
    )


def _normal_quantile(p: float) -> float:
    """Standard-normal inverse CDF (Acklam's rational approximation; no scipy).

    The a/b/c/d arrays are just Acklam's published magic constants for those fits.
    As written, the approximation has a maximum relative error of about 1.15 × 10⁻⁹
    across the whole interval — fine for confidence intervals, p-value work,
    and most statistics. (Acklam's full version adds one Halley refinement
    step using erfc to reach near machine precision; this implementation skips it,
    trading a bit of accuracy for not needing erfc.)

    One caveat worth knowing: it's only valid on the open interval (0, 1).
    At p = 0 or p = 1 you'd hit log(0) and get an infinity rather than a clean error,
    and it does no bounds checking on inputs outside [0, 1].
    """
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


# --------------------------------------------------------------------------- #
# Stratified document bootstrap
# --------------------------------------------------------------------------- #
def stratified_bootstrap_ci(
    point_fn,
    f_s: np.ndarray,
    y_s: np.ndarray,
    w_or_pi_s: np.ndarray,
    strata: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Percentile CI by resampling gold documents *within strata* with replacement.

    Resampling the document is what makes the CI valid: documents are the
    independent sampling unit, and resampling within stratum preserves the
    stratified design. ``point_fn(idx)`` must recompute the estimator from the
    bootstrap index array (the population term ``p_hat``/``N`` is fixed and closed
    over by the caller).
    """
    f_s = np.asarray(f_s)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    stratum_members = [np.where(strata == s)[0] for s in np.unique(strata)]

    samples = np.empty(n_boot, float)
    for b in range(n_boot):
        idx = np.concatenate(
            [m[rng.integers(0, len(m), len(m))] for m in stratum_members if len(m)]
        )
        samples[b] = point_fn(idx)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return np.nan, np.nan, samples
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi, samples


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
def estimate_prevalence(
    topic_id: object,
    f_pop: np.ndarray,
    f_s: np.ndarray,
    y_s: np.ndarray,
    pi_s: np.ndarray,
    strata: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    ci_method: str = "ppi_analytic",
) -> PrevalenceResult:
    """Full debiased prevalence for one topic: naive, ACC, PPI, and a 95% CI.

    Parameters
    ----------
    f_pop : (N,) surrogate labels over *every* document in the corpus.
    f_s, y_s : (n,) surrogate and gold-true labels on the labeled gold documents.
    pi_s : (n,) inclusion probability of each gold document (SRS → constant n/N).
    strata : (n,) stratum id per gold document (for the design-based variance).
    ci_method : headline CI — 'ppi_analytic' (default; design-based normal interval,
        best calibrated), 'ppi_boot', or 'acc_boot' (stratified document bootstrap).
    """
    f_pop = np.asarray(f_pop, float)
    f_s = np.asarray(f_s, float)
    y_s = np.asarray(y_s, float)
    pi_s = np.asarray(pi_s, float)
    strata = np.asarray(strata)

    n_docs = int(f_pop.shape[0])
    p_hat = float(f_pop.mean())
    w_s = 1.0 / pi_s
    tpr, fpr = weighted_confusion_rates(f_s, y_s, w_s)
    # Unweighted confusion-cell counts gate CI reliability (binding cell for a rare
    # topic is the false negatives, f=0 & Y=1).
    min_cell = int(
        min(
            ((f_s == 1) & (y_s == 1)).sum(),
            ((f_s == 1) & (y_s == 0)).sum(),
            ((f_s == 0) & (y_s == 1)).sum(),
            ((f_s == 0) & (y_s == 0)).sum(),
        )
    )
    pi_acc = acc_point(p_hat, f_s, y_s, w_s)
    pi_ppi = ppi_point(p_hat, f_s, y_s, pi_s, n_docs)

    se = np.nan
    if ci_method == "ppi_analytic":
        ci_low, ci_high, se = ppi_analytic_ci(
            pi_ppi, f_s, y_s, pi_s, strata, n_docs, alpha=alpha
        )
    elif ci_method == "acc_boot":
        ci_fn = lambda idx: acc_point(p_hat, f_s[idx], y_s[idx], w_s[idx])  # noqa: E731
        ci_low, ci_high, _ = stratified_bootstrap_ci(
            ci_fn, f_s, y_s, pi_s, strata, n_boot=n_boot, alpha=alpha, seed=seed
        )
    else:  # ppi_boot
        ci_fn = lambda idx: ppi_point(p_hat, f_s[idx], y_s[idx], pi_s[idx], n_docs)  # noqa: E731
        ci_low, ci_high, _ = stratified_bootstrap_ci(
            ci_fn, f_s, y_s, pi_s, strata, n_boot=n_boot, alpha=alpha, seed=seed
        )

    return PrevalenceResult(
        topic_id=topic_id,
        n_docs=n_docs,
        n_gold=int(f_s.shape[0]),
        n_gold_pos=int(y_s.sum()),
        naive=p_hat,
        tpr=tpr,
        fpr=fpr,
        pi_acc=pi_acc,
        pi_ppi=pi_ppi,
        ci_low=ci_low,
        ci_high=ci_high,
        se=se,
        ci_method=ci_method,
        min_cell=min_cell,
        reliable=min_cell >= MIN_CELL,
    )


# --------------------------------------------------------------------------- #
# Self-check: simulate a known world and confirm both estimators recover it.
# --------------------------------------------------------------------------- #
def _simulate(rng, n, pi_true, tpr, fpr, n_pos_gold, n_neg_gold):
    """One synthetic corpus + score-stratified gold draw. Returns the kwargs
    ``estimate_prevalence`` expects plus the ground-truth pi."""
    y = (rng.random(n) < pi_true).astype(int)
    f = np.where(y == 1, rng.random(n) < tpr, rng.random(n) < fpr).astype(int)

    pos_idx = np.where(f == 1)[0]
    neg_idx = np.where(f == 0)[0]
    n_pos = min(n_pos_gold, len(pos_idx))
    n_neg = min(n_neg_gold, len(neg_idx))
    s_pos = rng.choice(pos_idx, n_pos, replace=False)
    s_neg = rng.choice(neg_idx, n_neg, replace=False)
    s = np.concatenate([s_pos, s_neg])

    pi_s = np.concatenate(
        [np.full(n_pos, n_pos / len(pos_idx)), np.full(n_neg, n_neg / len(neg_idx))]
    )
    strata = np.concatenate([np.zeros(n_pos, int), np.ones(n_neg, int)])
    return dict(f_pop=f, f_s=f[s], y_s=y[s], pi_s=pi_s, strata=strata), pi_true


if __name__ == "__main__":
    # Bias + coverage check across regimes (common/rare topic, good/weak surrogate).
    rng = np.random.default_rng(0)
    scenarios = [
        ("common, good surrogate", dict(n=4000, pi_true=0.30, tpr=0.90, fpr=0.08)),
        ("rare, good surrogate", dict(n=8000, pi_true=0.03, tpr=0.85, fpr=0.03)),
        ("common, weak surrogate", dict(n=4000, pi_true=0.30, tpr=0.70, fpr=0.30)),
    ]
    for name, cfg in scenarios:
        acc_pts, ppi_pts = [], []
        cover = {"ppi_analytic": 0, "ppi_boot": 0}
        reps = 300
        for _ in range(reps):
            kw, pi_true = _simulate(rng, n_pos_gold=80, n_neg_gold=80, **cfg)
            seed = int(rng.integers(1e9))
            ra = estimate_prevalence("sim", **kw, ci_method="ppi_analytic", seed=seed)
            rb = estimate_prevalence(
                "sim", **kw, ci_method="ppi_boot", n_boot=600, seed=seed
            )
            acc_pts.append(ra.pi_acc)
            ppi_pts.append(ra.pi_ppi)
            cover["ppi_analytic"] += int(ra.ci_low <= pi_true <= ra.ci_high)
            cover["ppi_boot"] += int(rb.ci_low <= pi_true <= rb.ci_high)
        acc_pts = np.array(acc_pts)
        ppi_pts = np.array(ppi_pts)
        naive = cfg["pi_true"] * cfg["tpr"] + (1 - cfg["pi_true"]) * cfg["fpr"]
        print(f"\n[{name}]  true pi={cfg['pi_true']:.3f}  naive surrogate≈{naive:.3f}")
        print(
            f"   ACC mean={np.nanmean(acc_pts):.3f} (bias {np.nanmean(acc_pts) - cfg['pi_true']:+.3f})"
            f"   PPI mean={np.nanmean(ppi_pts):.3f} (bias {np.nanmean(ppi_pts) - cfg['pi_true']:+.3f})"
        )
        print(
            f"   95% CI coverage over {reps} reps: "
            f"analytic={cover['ppi_analytic'] / reps:.1%}  bootstrap={cover['ppi_boot'] / reps:.1%}"
        )
