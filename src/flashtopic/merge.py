"""Optional near-duplicate cluster merge (default-off in the pipeline).

K-means over-clusters by *wording*: the same topic, reworded, lands in several
clusters. Geometry can't separate "reworded duplicate" from "added content" at cluster
scale, but the **distinctive terms** can (e.g. one cluster keeps "tariff" while a
near-twin doesn't). So the merge runs on term-set overlap:

1. drop **generic terms** (present in > ``df_frac`` of clusters, plus English
   stopwords) so weak clusters don't glue distinct topics together;
2. **complete linkage** — a group forms only if *every* pair has Jaccard ≥ ``tau`` (a
   true duplicate clique), never a single-linkage chain.

This is opt-in: it loses fine distinctions (a near-twin that carried a rare term gets
folded away), so it's off by default — audit what it merges before trusting it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def complete_linkage_term_merge(
    term_lists: list[list[str]],
    *,
    tau: float = 0.5,
    df_frac: float = 0.05,
    top_terms: int = 12,
    extra_stopwords: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Merge clusters whose distinctive-term sets form a complete-linkage clique.

    Returns ``(labels, dropped_generic_terms)`` where ``labels[i]`` is cluster i's
    merged-topic id (compacted 0..T-1).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    K = len(term_lists)
    if K <= 1:
        return np.zeros(K, dtype=int), []

    raw = [[t.lower() for t in ts[:top_terms]] for ts in term_lists]
    df = Counter(t for s in raw for t in set(s))
    cap = max(4, int(df_frac * K))
    generic = {t for t, c in df.items() if c > cap}
    generic |= {t for t in df if t in ENGLISH_STOP_WORDS}
    generic |= {s.lower() for s in (extra_stopwords or set())}
    sets = [set(t for t in s if t not in generic) for s in raw]

    M = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            u = len(sets[i] | sets[j])
            M[i, j] = M[j, i] = (len(sets[i] & sets[j]) / u) if u else 0.0
    D = 1.0 - M
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="complete")
    labels = fcluster(Z, t=1.0 - tau, criterion="distance")
    remap = {v: i for i, v in enumerate(sorted(set(labels.tolist())))}
    compact = np.array([remap[v] for v in labels], dtype=np.int64)
    return compact, sorted(generic, key=lambda t: -df[t])
