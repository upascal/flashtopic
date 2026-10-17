"""End-to-end pilot on the synthetic world with the mock classifier (no network)."""

from __future__ import annotations

import pytest

from flashtopic.prevalence.classifier import MockClassifier
from flashtopic.prevalence.estimate import ppi_analytic_ci
from flashtopic.prevalence.pilot import (
    format_report, run_pilot_sync, synthetic_world, verdict, write_report,
)
from flashtopic.prevalence.prototypes import build_surrogate
from flashtopic.prevalence.sampling import design_sample


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    ctx, topics, y = synthetic_world(n_docs=300, seed=0)
    designs = {}
    for t in topics:
        sur = build_surrogate(ctx, t)
        des = design_sample(sur, n_pos=30, neg_bin_alloc=(30, 15, 10))
        designs[t.topic_id] = (des, y[t.topic_id][des.doc_ids])
    clf = MockClassifier()
    res = run_pilot_sync(ctx, topics, designs, clf, cache_dir=tmp_path_factory.mktemp("cache"))
    return ctx, designs, clf, res


def test_pilot_table_and_ses(pilot):
    ctx, designs, clf, res = pilot
    t = res.table
    assert set(t.topic_id) == set(designs) and len(res.qset.questions) == 6
    for _, r in t.iterrows():
        des, y = designs[r.topic_id]
        fits = res.topics[r.topic_id].fits
        for kind in ("memb", "det_lr", "lr", "presence_platt"):
            se = ppi_analytic_ci(0.0, fits[kind].f_gold, y, des.pi_s, des.strata, ctx.n_docs)[2]
            assert fits[kind].se_gold == pytest.approx(se)
        assert r.se_lr < r.se_memb and r.chosen in fits
        assert r.gold_needed_lr == pytest.approx(r.n_gold * (r.se_lr / r.se_memb) ** 2)
        assert 0 <= r.presence_brier <= 1 and r.n_missing_answers == 0


def test_classifier_called_once_per_gold_doc(pilot):
    ctx, designs, clf, res = pilot
    gold = {int(d) for des, _ in designs.values() for d in des.doc_ids}
    assert clf.calls == len(gold) == len(res.answers)


def test_report_and_verdict(pilot, tmp_path):
    _, _, _, res = pilot
    go, why = verdict(res.table)
    assert isinstance(go, bool) and "BUILD" in why
    md = format_report(res)
    assert "Verdict" in md and "Largest residuals" in md
    p_md, p_csv, p_q = write_report(res, out_dir=tmp_path)
    assert p_md.exists() and p_csv.exists() and p_q.exists()
