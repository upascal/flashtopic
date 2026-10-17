"""Feature assembly: stable names, deterministic block from unique sentences, classifier block."""

from __future__ import annotations

import numpy as np

from flashtopic.prevalence.classifier import Answer
from flashtopic.prevalence.data import RunContext
from flashtopic.prevalence.features import (
    DET_NAMES, assemble, classifier_feature_names, classifier_features, deterministic_features,
    presence_probability,
)
from flashtopic.prevalence.prototypes import build_surrogate
from flashtopic.prevalence.questions import TopicContext, build_question_set
from flashtopic.prevalence.topics import Topic


def _ctx():
    texts = ["strike picket union", "fare route timetable", "hotel dinner", "strike walkout"]
    uid_docs = [[0], [1], [0, 1], [0]]
    E = np.eye(4, dtype=np.float32)
    return RunContext(E=E, texts=texts, uid_docs=uid_docs, n_docs=2, doc_texts={})


def test_deterministic_features_from_unique_sentences_without_doc_texts():
    ctx = _ctx()
    t = Topic("T0", member_uids=[0, 3], label="strike", distinctive_terms=["strike", "picket", "zzz"])
    sur = build_surrogate(ctx, t)
    F = deterministic_features(ctx, t, sur, [0, 1])
    assert F.names == DET_NAMES and F.X.shape == (2, 5)
    d0, d1 = F.X
    assert d0[1] == 1 and d1[1] == 0  # f_memb
    assert d0[2] == 2 and d1[2] == 0  # member sentences in doc
    assert d0[4] == 2 and d1[4] == 0  # term hits (strike, picket; not zzz)
    assert d0[3] > d1[3] > 0  # log chars from the joined unique sentences


def test_classifier_block_names_and_missing_indicator():
    ctx = _ctx()
    ts = [Topic("T0", [0, 3], label="strike"), Topic("T1", [1], label="fares")]
    ctxs = {tid: TopicContext(meta_theme="transit", group_id="g", sibling_labels=("strike", "fares"))
            for tid in ("T0", "T1")}
    qs = build_question_set(ts, ctxs)
    names = classifier_feature_names(qs, "T0")
    assert names == ["presence.logit", "depth.expect", "depth.p_substantive", "parent.logit",
                     "sibling.p_self", "sibling.p_none", "missing"]

    answers = {
        0: [Answer("T0.presence", p_yes=0.9),
            Answer("T0.depth", dist=(0.0, 0.1, 0.2, 0.3, 0.4)),
            Answer("group:g.parent", p_yes=0.5),
            Answer("group:g.sibling", dist=(0.6, 0.3, 0.1))],
        1: [Answer("T0.presence", error="x")],  # everything else absent
    }
    sur = build_surrogate(ctx, ts[0])
    F = classifier_features(answers, qs, ts[0], [0, 1])
    r0, r1 = F.X
    assert r0[0] > 2.0  # logit(0.9)
    assert abs(r0[1] - (0.1 * 1 + 0.2 * 2 + 0.3 * 3 + 0.4 * 4) / 4) < 1e-9
    assert abs(r0[2] - np.log(0.9 / 0.1)) < 1e-6  # P(level >= 2) = 0.9
    assert r0[3] == 0.0  # logit(0.5)
    assert r0[4] > 0 > r0[5] and r0[6] == 0
    assert r1[6] == 4 and np.all(r1[:6] == 0)

    p = presence_probability(answers, "T0", [0, 1])
    assert p[0] == 0.9 and np.isnan(p[1])

    A = assemble(ctx, ts[0], sur, [0, 1], answers, qs)
    assert A.names == DET_NAMES + names and A.X.shape == (2, 12)
    assert A.select(["score", "missing"]).X[:, 1].tolist() == [0.0, 4.0]
