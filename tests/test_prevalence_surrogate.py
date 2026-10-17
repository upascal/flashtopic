"""Cross-fitted surrogate: no leakage, real gains when features inform, sound gating."""

from __future__ import annotations

import numpy as np
import pytest

from flashtopic.prevalence.estimate import estimate_prevalence
from flashtopic.prevalence.surrogate import (
    MIN_ESS, MIN_POS_FOR_LR, baseline_memb, choose_surrogate, design_se, doc_folds,
    fit_cross_fitted, kish_ess, platt_presence,
)


def _world(rng, n=6000, pi_true=0.2, n_feat=6, signal=2.0, tpr=0.8, fpr=0.1, sharp=False):
    """Latent-feature world: y ~ Bernoulli(sigmoid(z)) (or a near-deterministic cut of z
    when ``sharp``), X = informative + noise columns (+ f_memb, as in the real feature set),
    f_memb a noisy binarization. Returns population arrays."""
    z = rng.normal(size=n)
    if sharp:
        y = (z + rng.normal(scale=0.3, size=n) > 0.85).astype(int)
    else:
        y = (rng.random(n) < 1 / (1 + np.exp(-(signal * z - 1.5)))).astype(int)
    f_memb = np.where(y == 1, rng.random(n) < tpr, rng.random(n) < fpr).astype(int)
    X = np.column_stack([z + rng.normal(scale=0.5, size=n)]
                        + [rng.normal(size=n) for _ in range(n_feat - 2)] + [f_memb])
    return y, X, f_memb


def _design(rng, f_memb, rank, n_pos=60, alloc=(60, 25, 15)):
    pos = np.where(f_memb == 1)[0]
    neg = np.where(f_memb == 0)[0]
    order = neg[np.argsort(rank[neg])[::-1]]
    bins = np.array_split(order, len(alloc))
    ids, pi, strata = [], [], []
    for s, (pool, k) in enumerate([(pos, n_pos)] + list(zip(bins, alloc))):
        k = min(k, len(pool))
        sel = rng.choice(pool, k, replace=False)
        ids += sel.tolist()
        pi += [k / len(pool)] * k
        strata += [s] * k
    return np.array(ids), np.array(pi), np.array(strata)


def test_doc_folds_stable_across_topics_and_balanced():
    ids = np.arange(1000)
    f1 = doc_folds(ids, 5, seed=0)
    f2 = doc_folds(ids[::-1], 5, seed=0)[::-1]
    assert np.array_equal(f1, f2)
    assert set(f1) == set(range(5)) and np.bincount(f1).min() > 150
    assert not np.array_equal(f1, doc_folds(ids, 5, seed=1))


def test_design_se_matches_estimator_se():
    rng = np.random.default_rng(0)
    y, X, f_memb = _world(rng)
    ids, pi, strata = _design(rng, f_memb, X[:, 0])
    se = design_se(f_memb[ids], y[ids], pi, strata, len(y))
    r = estimate_prevalence("t", f_memb, f_memb[ids], y[ids], pi, strata)
    assert se == pytest.approx(r.se)


def test_no_leakage_pure_noise_features_do_not_shrink_se():
    """Out-of-fold predictions from pure noise can't beat a constant; an in-sample fit would."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    y, X, f_memb = _world(rng)
    noise = rng.normal(size=(len(y), 20))
    ids, pi, strata = _design(rng, f_memb, X[:, 0])
    fit = fit_cross_fitted("t", noise[ids], y[ids], pi, strata, ids, n_docs=len(y))
    const = np.full(len(ids), np.average(y[ids], weights=1 / pi))
    se_const = design_se(const, y[ids], pi, strata, len(y))
    # never optimistic: OOF noise predictions are a (slightly noisier) constant
    assert 0.85 * se_const <= fit.se_gold <= 1.6 * se_const

    # the leak: same features, in-sample, lightly regularized → optimistic residuals
    clf = LogisticRegression(C=100.0, max_iter=2000).fit(noise[ids], y[ids], sample_weight=1 / pi)
    se_leak = design_se(clf.predict_proba(noise[ids])[:, 1], y[ids], pi, strata, len(y))
    assert se_leak < 0.9 * fit.se_gold


def test_informative_features_beat_membership_and_f_pop_is_fold_mean():
    rng = np.random.default_rng(2)
    y, X, f_memb = _world(rng, sharp=True, tpr=0.7, fpr=0.15)
    ids, pi, strata = _design(rng, f_memb, X[:, 0])
    base = baseline_memb("t", f_memb[ids], y[ids], pi, strata, len(y), f_memb)
    fit = fit_cross_fitted("t", X[ids], y[ids], pi, strata, ids, X, n_docs=len(y),
                           feature_names=[f"x{i}" for i in range(X.shape[1])])
    assert fit.se_gold < 0.85 * base.se_gold
    assert fit.n_folds == 5 and fit.f_pop.shape == (len(y),)
    assert fit.f_pop.min() >= 0 and fit.f_pop.max() <= 1
    assert fit.coef_table()[0][0] == "x0"  # the informative column carries the weight
    # the chosen candidate is the regression once enough positives are labeled
    assert choose_surrogate([base, fit]).kind == "lr"
    # PPI with the cross-fitted f is still unbiased
    r = estimate_prevalence("t", fit.f_pop, fit.f_gold, y[ids], pi, strata)
    assert abs(r.pi_ppi - y.mean()) < 3 * r.se + 0.01


def test_platt_and_min_pos_gate():
    rng = np.random.default_rng(3)
    y, X, f_memb = _world(rng, pi_true=0.02, signal=3.0)
    ids, pi, strata = _design(rng, f_memb, X[:, 0])
    p = 1 / (1 + np.exp(-X[:, 0]))
    pl = platt_presence("t", p[ids], y[ids], pi, strata, ids, p, n_docs=len(y))
    assert pl.kind == "presence_platt" and pl.feature_names == ["presence.logit"]
    lr = fit_cross_fitted("t", X[ids], y[ids], pi, strata, ids, n_docs=len(y))
    few = choose_surrogate([lr, pl], min_pos=10**6)  # pretend too few positives
    assert few.kind == "presence_platt"
    assert MIN_POS_FOR_LR == 10


def test_ess_fallback_tempers_weights():
    rng = np.random.default_rng(4)
    y, X, f_memb = _world(rng)
    ids, pi, strata = _design(rng, f_memb, X[:, 0], n_pos=100, alloc=(3, 3, 60))
    assert kish_ess(1 / pi) < MIN_ESS
    fit = fit_cross_fitted("t", X[ids], y[ids], pi, strata, ids, n_docs=len(y))
    assert fit.weight_power == 0.5 and fit.ess == pytest.approx(kish_ess((1 / pi) ** 0.5))


@pytest.mark.slow
def test_cross_fitted_ppi_coverage():
    """Coverage of the analytic 95% CI when f is the cross-fitted regression (60 reps)."""
    rng = np.random.default_rng(5)
    hits, reps = 0, 60
    for _ in range(reps):
        y, X, f_memb = _world(rng, n=3000, sharp=True)
        ids, pi, strata = _design(rng, f_memb, X[:, 0])
        fit = fit_cross_fitted("t", X[ids], y[ids], pi, strata, ids, X, n_docs=len(y))
        r = estimate_prevalence("t", fit.f_pop, fit.f_gold, y[ids], pi, strata)
        hits += int(r.ci_low <= y.mean() <= r.ci_high)
    assert hits / reps >= 0.85
