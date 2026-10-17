"""Assemble pilot results and the human spot-check that validates the LLM gold.

Two jobs:

* :func:`build_table` / :func:`save_table` — one row per topic comparing the naive
  (uncorrected surrogate) prevalence against the debiased PPI/ACC estimate + CI, with
  the reliability flag. This is the publication-grade table (plan §7.3, scoped).
* :func:`export_spotcheck` / :func:`spotcheck_agreement` — sample judged documents
  into a sheet for the user to blind-label, then measure LLM↔human agreement (pilot
  step 5). If agreement is high the LLM gold stands; if not, the codebook is tightened
  before any number is trusted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flashtopic.prevalence.estimate import PrevalenceResult
from flashtopic.prevalence.judge import Verdict
from flashtopic.prevalence.prototypes import TopicSurrogate
from flashtopic.prevalence.sampling import SampleDesign

# Outputs land in the caller's working directory, never inside the installed package.
_OUT_DIR = Path.cwd() / "flashtopic_output"


def result_row(
    topic_label: str,
    kind: str,
    sur: TopicSurrogate,
    res: PrevalenceResult,
    llm_human_agreement: float | None = None,
) -> dict:
    return {
        "topic_id": res.topic_id,
        "kind": kind,
        "label": topic_label,
        "n_doc": res.n_docs,
        "naive_prev": round(sur.naive, 4),
        "pi_acc": round(res.pi_acc, 4) if not np.isnan(res.pi_acc) else np.nan,
        "pi_ppi": round(res.pi_ppi, 4),
        "ci_low": round(res.ci_low, 4),
        "ci_high": round(res.ci_high, 4),
        "correction": round(res.pi_ppi - sur.naive, 4),
        "ci_method": res.ci_method,
        "tpr": round(res.tpr, 3) if not np.isnan(res.tpr) else np.nan,
        "fpr": round(res.fpr, 3) if not np.isnan(res.fpr) else np.nan,
        "n_gold": res.n_gold,
        "n_gold_pos": res.n_gold_pos,
        "min_cell": res.min_cell,
        "reliable": res.reliable,
        "llm_human_agreement": llm_human_agreement,
    }


def build_table(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "topic_id", "kind", "label", "n_doc", "naive_prev", "pi_ppi", "ci_low",
        "ci_high", "correction", "pi_acc", "ci_method", "tpr", "fpr", "n_gold",
        "n_gold_pos", "min_cell", "reliable", "llm_human_agreement",
    ]
    return pd.DataFrame(rows)[cols].sort_values("pi_ppi", ascending=False).reset_index(drop=True)


def save_table(df: pd.DataFrame, name: str = "prevalence_pilot") -> tuple[Path | None, Path]:
    """Write CSV always; Parquet too if an engine (pyarrow/fastparquet) is installed."""
    _OUT_DIR.mkdir(exist_ok=True)
    csv = _OUT_DIR / f"{name}.csv"
    df.to_csv(csv, index=False)
    pq: Path | None = _OUT_DIR / f"{name}.parquet"
    try:
        df.to_parquet(pq)
    except ImportError:
        pq = None  # parquet engine absent — CSV is enough for the pilot
    return pq, csv


def format_table(df: pd.DataFrame) -> str:
    """Compact console view: naive → debiased [CI], correction, reliability."""
    lines = [
        f"{'topic':<7}{'label':<40}{'naive':>7}{'debiased':>10}"
        f"{'95% CI':>16}{'Δ':>8}  flag"
    ]
    for _, r in df.iterrows():
        flag = "" if r["reliable"] else "⚠ recall-limited"
        ci = f"[{r['ci_low']:.2f},{r['ci_high']:.2f}]"
        lines.append(
            f"{str(r['topic_id']):<7}{str(r['label'])[:38]:<40}"
            f"{r['naive_prev']:>7.2f}{r['pi_ppi']:>10.2f}{ci:>16}"
            f"{r['correction']:>+8.2f}  {flag}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Human spot-check (validate the LLM gold)
# --------------------------------------------------------------------------- #
def export_spotcheck(
    sheets: list[tuple[SampleDesign, list[Verdict]]],
    doc_texts: dict[int, str],
    n_per_topic: int = 20,
    seed: int = 0,
    name: str = "spotcheck",
) -> Path:
    """Write a blind spot-check sheet: judged docs with a blank ``human`` column.

    Stratified across each topic's gold strata so the check covers positives and the
    borderline negatives, not just the easy cases.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for design, verdicts in sheets:
        vmap = {v.doc_id: v for v in verdicts}
        # spread the check across strata
        per_stratum = max(1, n_per_topic // max(1, len(np.unique(design.strata))))
        picks = []
        for s in np.unique(design.strata):
            pool = design.doc_ids[design.strata == s]
            picks += list(rng.choice(pool, min(per_stratum, len(pool)), replace=False))
        for d in picks[:n_per_topic]:
            v = vmap[int(d)]
            rows.append({
                "topic_id": design.topic_id,
                "doc_id": int(d),
                "judge": int(v.discusses),
                "human": "",  # ← fill in: 1 if the document substantively discusses it, else 0
                "judge_rationale": v.rationale,
                "excerpt": doc_texts[int(d)][:300].replace("\n", " "),
            })
    _OUT_DIR.mkdir(exist_ok=True)
    path = _OUT_DIR / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def spotcheck_agreement(name: str = "spotcheck") -> dict:
    """Read the filled sheet → overall + per-topic LLM↔human agreement."""
    df = pd.read_csv(_OUT_DIR / f"{name}.csv")
    df = df[df["human"].apply(lambda x: str(x).strip() in {"0", "1"})].copy()
    if df.empty:
        return {"n": 0, "overall": None, "per_topic": {}}
    df["human"] = df["human"].astype(int)
    df["agree"] = (df["human"] == df["judge"]).astype(int)
    per_topic = df.groupby("topic_id")["agree"].mean().round(3).to_dict()
    return {"n": len(df), "overall": round(df["agree"].mean(), 3), "per_topic": per_topic}
