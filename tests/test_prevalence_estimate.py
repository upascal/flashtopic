"""Unit tests for the debiased prevalence estimators (notebooks/prevalence_estimation/estimate.py).

Pure-numpy, no corpus needed. Validates the property that matters: both estimators
recover a known prevalence even when the cheap surrogate is biased, and the
sparse-cell flag fires when a confusion cell is too thin to trust the CI.
"""

from __future__ import annotations


import numpy as np
import pytest


from flashtopic.prevalence.estimate import (  # noqa: E402
    acc_point,
    estimate_prevalence,
    ppi_point,
    weighted_confusion_rates,
    _simulate,
)


def _draw(rng, **cfg):
    return _simulate(rng, n_pos_gold=120, n_neg_gold=120, **cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(n=4000, pi_true=0.30, tpr=0.90, fpr=0.08),  # common, good surrogate
        dict(n=4000, pi_true=0.30, tpr=0.70, fpr=0.30),  # common, weak surrogate
        dict(n=5000, pi_true=0.15, tpr=0.80, fpr=0.10),  # mid prevalence
    ],
)
def test_estimators_are_unbiased(cfg):
    """Across 200 replications the mean ACC/PPI estimate sits within MC error of the
    truth — the debiasing works regardless of surrogate quality."""
    rng = np.random.default_rng(0)
    acc, ppi = [], []
    for _ in range(200):
        kw, pi_true = _draw(rng, **cfg)
        r = estimate_prevalence("t", **kw, ci_method="ppi_analytic", seed=int(rng.integers(1e9)))
        acc.append(r.pi_acc)
        ppi.append(r.pi_ppi)
    assert abs(np.nanmean(acc) - cfg["pi_true"]) < 0.02
    assert abs(np.nanmean(ppi) - cfg["pi_true"]) < 0.02


def test_naive_is_biased_when_surrogate_is():
    """Sanity: the uncorrected surrogate rate is the biased number we are fixing."""
    rng = np.random.default_rng(3)
    cfg = dict(n=4000, pi_true=0.30, tpr=0.70, fpr=0.30)
    kw, pi_true = _draw(rng, **cfg)
    r = estimate_prevalence("t", **kw, seed=1)
    # naive overshoots (FPR inflates it); debiased lands near the truth.
    assert r.naive > pi_true + 0.05
    assert abs(r.pi_ppi - pi_true) < 0.05


def test_ppi_equals_acc_under_srs_binary_surrogate():
    """With a simple random sample and a binary surrogate, PPI and ACC are close."""
    rng = np.random.default_rng(7)
    n = 6000
    y = (rng.random(n) < 0.2).astype(int)
    f = np.where(y == 1, rng.random(n) < 0.8, rng.random(n) < 0.1).astype(int)
    idx = rng.choice(n, 600, replace=False)
    pi_s = np.full(600, 600 / n)
    r = estimate_prevalence(
        "t", f_pop=f, f_s=f[idx], y_s=y[idx], pi_s=pi_s, strata=np.zeros(600, int), seed=0
    )
    assert abs(r.pi_acc - r.pi_ppi) < 0.03


def test_sparse_cell_flag_fires_for_rare_topic():
    """A rare topic with a uniform negative sample starves the false-negative cell —
    the result must mark itself unreliable rather than report a falsely tight CI."""
    rng = np.random.default_rng(11)
    cfg = dict(n=8000, pi_true=0.02, tpr=0.85, fpr=0.02)
    kw, _ = _simulate(rng, n_pos_gold=60, n_neg_gold=60, **cfg)
    r = estimate_prevalence("t", **kw, seed=0)
    assert r.min_cell < 5
    assert r.reliable is False


def test_weighted_rates_undo_oversampling():
    """HT weights recover population TPR/FPR from a stratified (oversampled) gold set.

    Averaged over draws — a single draw's false-negative cell is sparse and noisy,
    but the reweighting is unbiased in expectation.
    """
    rng = np.random.default_rng(5)
    n = 10000
    y = (rng.random(n) < 0.1).astype(int)
    f = np.where(y == 1, rng.random(n) < 0.75, rng.random(n) < 0.05).astype(int)
    pos, neg = np.where(f == 1)[0], np.where(f == 0)[0]
    tpr_true, fpr_true = f[y == 1].mean(), f[y == 0].mean()

    tprs, fprs = [], []
    for _ in range(150):
        s_pos = rng.choice(pos, 200, replace=False)
        s_neg = rng.choice(neg, 200, replace=False)
        s = np.concatenate([s_pos, s_neg])
        w = np.concatenate([np.full(200, len(pos) / 200), np.full(200, len(neg) / 200)])
        tpr, fpr = weighted_confusion_rates(f[s], y[s], w)
        tprs.append(tpr)
        fprs.append(fpr)
    assert abs(np.mean(tprs) - tpr_true) < 0.02
    assert abs(np.mean(fprs) - fpr_true) < 0.02
