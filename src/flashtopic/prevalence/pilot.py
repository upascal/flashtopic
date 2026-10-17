"""The cheap pilot: does a classifier-fed surrogate tighten the prevalence CI?

The analytic PPI standard error depends **only on the gold rows' residuals** ``f - Y``
(``estimate.ppi_analytic_ci``), so the question "how much would the CI shrink" is answered
without scoring the corpus: run the decision model on the gold documents only, fit each
candidate surrogate cross-fitted, and compare the design SE each one yields at the *same*
gold size.

Per topic the pilot fits three candidates (plus the raw-presence recalibration):

* ``memb``   — today's membership indicator
* ``det_lr`` — regression on deterministic features only (isolates the classifier's value)
* ``lr``     — regression on deterministic + classifier features

and reports SE / CI half-width for each, the ratio, the gold size ``lr`` would need to match
``memb``'s CI, the classifier's zero-shot presence AUC/Brier against the judge, which
questions carry weight, and whether ranking negatives by presence probability finds more
gold positives than the cosine score does (the stratification question). The largest
residuals are dumped for an LLM-assisted look — input to a *next* question set, never to
this one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from flashtopic.prevalence.classifier import Answer, DocClassifier, classify_docs
from flashtopic.prevalence.data import RunContext
from flashtopic.prevalence.estimate import _normal_quantile
from flashtopic.prevalence.features import (
    FeatureMatrix, assemble, deterministic_features, presence_probability,
)
from flashtopic.prevalence.prototypes import TopicSurrogate, build_surrogate
from flashtopic.prevalence.questions import QuestionSet, TopicContext, build_question_set
from flashtopic.prevalence.sampling import SampleDesign
from flashtopic.prevalence.surrogate import (
    FittedSurrogate, baseline_memb, choose_surrogate, fit_cross_fitted, platt_presence,
)
from flashtopic.prevalence.topics import Topic

_OUT_DIR = Path.cwd() / "flashtopic_output"
_Z95 = _normal_quantile(0.975)


@dataclass
class PilotRow:
    topic_id: str
    label: str
    n_gold: int
    n_gold_pos: int
    naive: float
    se_memb: float
    se_det_lr: float
    se_presence: float
    se_lr: float
    ci_half_memb: float
    ci_half_lr: float
    ratio_lr_vs_memb: float  # se_lr / se_memb  (< 1 is a win)
    ratio_lr_vs_det: float  # se_lr / se_det_lr (< 1 means the classifier adds value)
    gold_needed_lr: float  # gold size for lr to match memb's CI (≈ n_gold · ratio²)
    chosen: str
    ess: float
    presence_auc: float
    presence_brier: float
    pos_in_top_neg_bin_cosine: int
    pos_in_top_neg_bin_presence: int
    n_missing_answers: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class TopicPilot:
    row: PilotRow
    fits: dict[str, FittedSurrogate]
    residual_dump: list[dict] = field(default_factory=list)


@dataclass
class PilotResult:
    table: pd.DataFrame
    topics: dict[str, TopicPilot]
    qset: QuestionSet
    answers: dict[int, list[Answer]]
    classifier_name: str


# --------------------------------------------------------------------------- #
# Per-topic comparison (pure numpy/sklearn; no model calls)
# --------------------------------------------------------------------------- #
def _auc_brier(y, p) -> tuple[float, float]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    m = ~np.isnan(p)
    if m.sum() < 2 or len(set(y[m].tolist())) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y[m], p[m])), float(brier_score_loss(y[m], p[m]))


def _pos_in_top_third(y, f_memb, rank) -> int:
    """Among gold docs the membership predictor calls negative, how many true positives
    land in the top third by ``rank``? Higher = the ranking hunts false negatives better."""
    neg = np.where(f_memb == 0)[0]
    if len(neg) == 0:
        return 0
    r = np.where(np.isnan(rank[neg]), -np.inf, rank[neg])
    top = neg[np.argsort(r)[::-1][: max(1, len(neg) // 3)]]
    return int(y[top].sum())


def compare_on_gold(
    topic: Topic,
    design: SampleDesign,
    y: np.ndarray,
    n_docs: int,
    F_det: FeatureMatrix,
    F_all: FeatureMatrix | None,
    p_presence: np.ndarray | None,
    sur: TopicSurrogate,
    *,
    seed: int = 0,
    n_dump: int = 10,
    doc_texts: dict[int, str] | None = None,
) -> TopicPilot:
    y = np.asarray(y, int)
    pi, strata, ids = design.pi_s, design.strata, design.doc_ids
    f_memb = sur.f_memb[ids].astype(float)

    fits: dict[str, FittedSurrogate] = {}
    fits["memb"] = baseline_memb(topic.topic_id, f_memb, y, pi, strata, n_docs)
    fits["det_lr"] = fit_cross_fitted(topic.topic_id, F_det.X, y, pi, strata, ids, n_docs=n_docs,
                                      feature_names=F_det.names, seed=seed, kind="det_lr")
    if p_presence is not None and np.isfinite(p_presence).any():
        p_filled = np.where(np.isnan(p_presence), 0.5, p_presence)
        fits["presence_platt"] = platt_presence(topic.topic_id, p_filled, y, pi, strata, ids,
                                                n_docs=n_docs, seed=seed)
    if F_all is not None:
        fits["lr"] = fit_cross_fitted(topic.topic_id, F_all.X, y, pi, strata, ids, n_docs=n_docs,
                                      feature_names=F_all.names, seed=seed, kind="lr")

    chosen = choose_surrogate(list(fits.values()))
    se_memb = fits["memb"].se_gold
    se_lr = fits["lr"].se_gold if "lr" in fits else float("nan")
    se_det = fits["det_lr"].se_gold
    se_pres = fits["presence_platt"].se_gold if "presence_platt" in fits else float("nan")
    ratio = se_lr / se_memb if se_memb > 0 else float("nan")
    auc, brier = _auc_brier(y, p_presence) if p_presence is not None else (float("nan"),) * 2
    n_missing = 0
    if F_all is not None and "missing" in F_all.names:
        n_missing = int(F_all.X[:, F_all.names.index("missing")].sum())

    row = PilotRow(
        topic_id=topic.topic_id, label=topic.label or "", n_gold=int(len(y)),
        n_gold_pos=int(y.sum()), naive=float(sur.naive),
        se_memb=se_memb, se_det_lr=se_det, se_presence=se_pres, se_lr=se_lr,
        ci_half_memb=_Z95 * se_memb, ci_half_lr=_Z95 * se_lr,
        ratio_lr_vs_memb=ratio,
        ratio_lr_vs_det=se_lr / se_det if se_det > 0 else float("nan"),
        gold_needed_lr=len(y) * ratio**2 if np.isfinite(ratio) else float("nan"),
        chosen=chosen.kind, ess=fits["lr"].ess if "lr" in fits else fits["det_lr"].ess,
        presence_auc=auc, presence_brier=brier,
        pos_in_top_neg_bin_cosine=_pos_in_top_third(y, f_memb, sur.score[ids]),
        pos_in_top_neg_bin_presence=(_pos_in_top_third(y, f_memb, p_presence)
                                     if p_presence is not None else 0),
        n_missing_answers=n_missing,
    )

    best = fits.get("lr", fits["det_lr"])
    resid = np.abs(best.f_gold - y)
    dump = []
    for i in np.argsort(resid)[::-1][:n_dump]:
        d = int(ids[i])
        dump.append({
            "doc_id": d, "y": int(y[i]), "f": round(float(best.f_gold[i]), 3),
            "f_memb": int(f_memb[i]), "score": round(float(sur.score[d]), 3),
            "p_presence": (None if p_presence is None or np.isnan(p_presence[i])
                           else round(float(p_presence[i]), 3)),
            "stratum": int(strata[i]),
            "excerpt": ((doc_texts or {}).get(d, "")[:300].replace("\n", " ")),
        })
    return TopicPilot(row=row, fits=fits, residual_dump=dump)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _gold_texts(ctx: RunContext, doc_ids) -> dict[int, str]:
    out = {}
    for d in doc_ids:
        d = int(d)
        t = ctx.doc_texts.get(d)
        out[d] = t if t is not None else "\n".join(ctx.texts[u] for u in ctx.doc_uids.get(d, []))
    return out


async def run_pilot(
    ctx: RunContext,
    topics: list[Topic],
    designs: dict[str, tuple[SampleDesign, np.ndarray]],
    clf: DocClassifier,
    *,
    ctx_by_topic: dict[str, TopicContext] | None = None,
    codebooks: dict | None = None,
    surrogates: dict[str, TopicSurrogate] | None = None,
    cache_dir: Path | None = None,
    concurrency: int = 8,
    seed: int = 0,
    on_progress=None,
) -> PilotResult:
    """Classify every gold document once (all topics' questions), then compare per topic."""
    from flashtopic.prevalence.classifier import _DEFAULT_CACHE_DIR

    topics = [t for t in topics if t.topic_id in designs]
    qset = build_question_set(topics, ctx_by_topic, codebooks)
    gold_ids = sorted({int(d) for des, _ in designs.values() for d in des.doc_ids})
    texts = _gold_texts(ctx, gold_ids)
    answers = await classify_docs(clf, texts, qset.questions, concurrency=concurrency,
                                  cache_dir=cache_dir or _DEFAULT_CACHE_DIR, on_progress=on_progress)

    surrogates = surrogates or {}
    per_topic: dict[str, TopicPilot] = {}
    for t in topics:
        design, y = designs[t.topic_id]
        sur = surrogates.get(t.topic_id) or build_surrogate(ctx, t)
        F_det = deterministic_features(ctx, t, sur, design.doc_ids)
        F_all = assemble(ctx, t, sur, design.doc_ids, answers, qset)
        p_pres = presence_probability(answers, t.topic_id, design.doc_ids)
        per_topic[t.topic_id] = compare_on_gold(t, design, np.asarray(y), ctx.n_docs, F_det, F_all,
                                                p_pres, sur, seed=seed, doc_texts=texts)

    table = pd.DataFrame([tp.row.to_dict() for tp in per_topic.values()])
    return PilotResult(table=table, topics=per_topic, qset=qset, answers=answers,
                       classifier_name=clf.name)


def run_pilot_sync(*a, **kw) -> PilotResult:
    import asyncio

    return asyncio.run(run_pilot(*a, **kw))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def verdict(table: pd.DataFrame) -> tuple[bool, str]:
    """Decision rule from the plan: build it only if lr beats memb on most topics AND beats
    the deterministic-only regression (so the classifier itself is what helps)."""
    t = table.dropna(subset=["ratio_lr_vs_memb", "ratio_lr_vs_det"])
    if t.empty:
        return False, "no topic had a fitted classifier surrogate"
    beats_memb = (t["ratio_lr_vs_memb"] < 0.9).mean()
    beats_det = (t["ratio_lr_vs_det"] < 0.95).mean()
    go = bool(beats_memb > 0.5 and beats_det > 0.5)
    return go, (f"lr beat memb by ≥10% on {beats_memb:.0%} of topics and beat the "
                f"deterministic-only regression by ≥5% on {beats_det:.0%} → "
                + ("BUILD" if go else "DO NOT BUILD (yet)"))


def format_report(res: PilotResult) -> str:
    t = res.table
    go, why = verdict(t)
    lines = ["# Jev-as-feature-extractor pilot", "",
             f"Classifier: `{res.classifier_name}` · question set `{res.qset.hash}` "
             f"({len(res.qset.questions)} questions) · gold docs classified: {len(res.answers)}",
             "", f"**Verdict:** {why}", "",
             "The SE below is the design-based PPI standard error implied by each surrogate's "
             "out-of-fold residuals on the gold rows — the only thing a better surrogate can "
             "change. `ratio` < 1 is a tighter CI at the same gold size; `gold_needed` is the "
             "gold size `lr` needs to match today's CI.", "",
             "| topic | label | n_gold | pos | naive | se memb | se det_lr | se presence | se lr | "
             "ratio vs memb | ratio vs det | gold needed | chosen | ESS | AUC | Brier | "
             "pos in top neg bin (cos → jev) | missing |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(
            f"| {r.topic_id} | {str(r.label)[:32]} | {r.n_gold} | {r.n_gold_pos} | {r.naive:.3f} | "
            f"{r.se_memb:.4f} | {r.se_det_lr:.4f} | {r.se_presence:.4f} | {r.se_lr:.4f} | "
            f"{r.ratio_lr_vs_memb:.2f} | {r.ratio_lr_vs_det:.2f} | {r.gold_needed_lr:.0f} | "
            f"{r.chosen} | {r.ess:.0f} | {r.presence_auc:.2f} | {r.presence_brier:.3f} | "
            f"{r.pos_in_top_neg_bin_cosine} → {r.pos_in_top_neg_bin_presence} | {r.n_missing_answers} |")
    lines += ["", "## Which questions carry weight (standardized LR coefficients, mean over folds)", ""]
    for tid, tp in res.topics.items():
        fit = tp.fits.get("lr")
        if fit is None or fit.coef_mean is None:
            continue
        coefs = ", ".join(f"{n}={c:+.2f}" for n, c in fit.coef_table()[:8])
        lines.append(f"- **{tid}** ({tp.row.label[:40]}): {coefs}")
    lines += ["", "## Largest residuals (input to the *next* question set — not this one)", ""]
    for tid, tp in res.topics.items():
        lines.append(f"### {tid} — {tp.row.label[:60]}")
        lines.append("")
        lines.append("| doc | y | f | f_memb | score | p_presence | stratum | excerpt |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for d in tp.residual_dump:
            lines.append(f"| {d['doc_id']} | {d['y']} | {d['f']} | {d['f_memb']} | {d['score']} | "
                         f"{d['p_presence']} | {d['stratum']} | {d['excerpt'][:160]} |")
        lines.append("")
    return "\n".join(lines)


def write_report(res: PilotResult, name: str = "jev_pilot", out_dir: Path = _OUT_DIR) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{name}.md"
    csv = out_dir / f"{name}.csv"
    js = out_dir / f"{name}_questions.json"
    md.write_text(format_report(res))
    res.table.to_csv(csv, index=False)
    js.write_text(json.dumps([q.__dict__ for q in res.qset.questions], indent=2, default=list))
    return md, csv, js


# --------------------------------------------------------------------------- #
# Self-check on a synthetic world (no network): MockClassifier + synthetic gold
# --------------------------------------------------------------------------- #
def synthetic_world(n_docs: int = 400, seed: int = 0):
    """A tiny corpus of keyword documents with a known latent topic label per doc.

    Returns ``(ctx, topics, y_by_topic)`` where ``y_by_topic[topic_id]`` is the true
    per-document "discusses" label. Membership is deliberately noisy so there is something
    for a better surrogate to recover.
    """
    from flashtopic.prevalence.topics import topics_from_labels

    rng = np.random.default_rng(seed)
    themes = {
        0: ("strike", ["strike", "picket", "union", "walkout", "wages", "negotiate"]),
        1: ("fares", ["fare", "route", "timetable", "rider", "pass", "discount"]),
        2: ("advertising", ["billboard", "campaign", "slogan", "brand", "poster", "jingle"]),
    }
    filler = ["weather", "coffee", "park", "crowd", "lunch", "hotel", "dinner", "radio"]
    texts, uid_docs, doc_texts = [], [], {}
    y = {f"T{c}": np.zeros(n_docs, int) for c in themes}
    labels = []
    for d in range(n_docs):
        sents = []
        for c, (_, words) in themes.items():
            on = rng.random() < 0.25
            if on:
                y[f"T{c}"][d] = 1
                k = rng.integers(2, 4)
                for _ in range(k):
                    sents.append((c, " ".join(rng.choice(words, 3)) + f" {d}"))
            elif rng.random() < 0.15:  # passing mention: one topic word, does NOT count
                sents.append((-1, rng.choice(words) + " " + " ".join(rng.choice(filler, 2)) + f" {d}"))
        for _ in range(rng.integers(1, 3)):
            sents.append((-1, " ".join(rng.choice(filler, 3)) + f" {d}"))
        for c, s in sents:
            texts.append(s)
            uid_docs.append([d])
            # membership: passing mentions get assigned to the topic 60% of the time (noise)
            if c == -1:
                c = int(rng.integers(0, 3)) if rng.random() < 0.6 and any(
                    w in s for ws in [t[1] for t in themes.values()] for w in ws) else 3
            labels.append(c)
        doc_texts[d] = "\n".join(s for _, s in sents)
    E = rng.normal(size=(len(texts), 8)).astype(np.float32)
    for i, c in enumerate(labels):  # give member sentences a shared direction
        if c < 3:
            E[i, c] += 3.0
    ctx = RunContext(E=E, texts=texts, uid_docs=uid_docs, n_docs=n_docs, doc_texts=doc_texts)
    topics = topics_from_labels(np.array(labels),
                                label_text={c: themes[c][0] for c in themes},
                                distinctive_terms={c: themes[c][1] for c in themes})
    topics = [t for t in topics if t.topic_id in y]
    return ctx, topics, y


if __name__ == "__main__":
    import tempfile

    from flashtopic.prevalence.classifier import MockClassifier
    from flashtopic.prevalence.sampling import design_sample

    ctx, topics, y = synthetic_world()
    designs = {}
    for t in topics:
        sur = build_surrogate(ctx, t)
        des = design_sample(sur, n_pos=40, neg_bin_alloc=(40, 20, 10))
        designs[t.topic_id] = (des, y[t.topic_id][des.doc_ids])
    with tempfile.TemporaryDirectory() as tmp:
        res = run_pilot_sync(ctx, topics, designs, MockClassifier(), cache_dir=Path(tmp))
    print(format_report(res))
