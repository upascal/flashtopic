"""Hierarchical contrastive labeling — the driver for ``label.py``'s group labeler.

``label.py`` ships :func:`label_group` / :func:`label_level_all` /
:func:`batch_by_similarity` but no orchestrator that turns a clustering's tiers into
contrastive calls. This module is that driver.

Given per-unique label arrays (coarse → fine), the child→parent maps, and per-cluster
terms + exemplars, it labels each tier's clusters **against their siblings** in one pass:
near-duplicate sibling labels become differentiated ones, and each group yields a
``parent_theme``. A tier's own label comes from the pass where it is the *child*; the
parent tier's ``parent_theme`` is captured as a cross-signal. The coarsest tier is
labeled as the children of a single synthetic root group, so flat clusterings (one
level) work too.

No DB/Workspace coupling — plain NumPy in, plain dicts out — so any store can drive it.
"""

from __future__ import annotations

import numpy as np

from flashtopic.label import (
    ChildEvidence,
    _group_hash,
    batch_by_similarity,
    label_level_all,
)

DEFAULT_BATCH = 8


def _centroids_by_id(E: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(k, d) L2-normalized centroid per cluster id, for similarity batching."""
    from flashtopic.cluster.levels import _centroids

    return _centroids(np.asarray(E, dtype=np.float32), np.asarray(labels))


async def label_levels(
    *,
    E: np.ndarray,
    level_labels,                # list of (n_uniques,) int arrays, coarse → fine
    parents,                     # list[dict]: parents[i] maps a level-(i+1) id → level-i id
    terms_by_tier,               # {tier_idx: {cluster_id: [terms]}}
    exemplars_by_tier,           # {tier_idx: {cluster_id: [exemplar lines]}}
    corpus_topic: str = "",
    model: str,
    level_names,                 # list[str] coarse → fine, len == len(level_labels)
    root_name: str = "corpus",
    batch_size: int = DEFAULT_BATCH,
    concurrency: int = 8,
    persistent: dict | None = None,   # {group_hash: cached entry} for cross-run reuse
):
    """Contrastively label every tier's clusters against their siblings.

    Returns ``(labels_by_tier, entries)`` where
    ``labels_by_tier[tier_idx][cluster_id] = {"label", "differs", "parent_id",
    "parent_theme", "parent_unifies"}`` and ``entries`` is the flat list of cache
    entries produced (each carries its ``evidence_hash``) so the caller can persist them.
    """
    persistent = persistent or {}
    n_levels = len(level_labels)
    out: dict[int, dict[int, dict]] = {ci: {} for ci in range(n_levels)}
    entries: list[dict] = []

    for ci in range(n_levels):
        labels = np.asarray(level_labels[ci])
        child_ids = sorted(int(c) for c in set(labels.tolist()))
        if not child_ids:
            continue

        # Group children by parent (coarsest tier → one synthetic root group).
        if ci == 0:
            groups: dict = {None: child_ids}
            parent_name = root_name
        else:
            pmap = parents[ci - 1]
            groups = {}
            for cid in child_ids:
                groups.setdefault(int(pmap[cid]), []).append(cid)
            parent_name = level_names[ci - 1]
        child_level = level_names[ci]

        centroids = _centroids_by_id(E, labels)

        # Build one contrastive call per batch; oversized groups split by similarity so
        # the most-confusable siblings land in the same call.
        calls: dict[str, dict] = {}
        call_meta: dict[str, dict] = {}
        for pid, kids in groups.items():
            if len(kids) > batch_size:
                med = np.asarray([centroids[c] for c in kids], dtype=np.float32)
                batches = [[int(x) for x in b]
                           for b in batch_by_similarity([str(c) for c in kids], med, n=batch_size)]
            else:
                batches = [list(kids)]
            for bi, batch in enumerate(batches):
                key = f"{ci}:{pid}:{bi}"
                children = []
                ref_to_cid: dict[str, int] = {}
                for cid in batch:
                    ref = f"{ci}:{cid}"
                    ref_to_cid[ref] = cid
                    children.append(
                        ChildEvidence(
                            ref=ref,
                            terms=list(terms_by_tier.get(ci, {}).get(cid, [])),
                            evidence_lines=list(exemplars_by_tier.get(ci, {}).get(cid, [])),
                            subject_terms=(
                                list(terms_by_tier.get(ci - 1, {}).get(pid, []))
                                if ci > 0 and pid is not None else []
                            ),
                        )
                    )
                # Keep batches of the same parent distinct from each other.
                neighbor_terms = None
                if len(batches) > 1:
                    others = [c for ob in batches if ob is not batch for c in ob]
                    neighbor_terms = [list(terms_by_tier.get(ci, {}).get(c, [])) for c in others]
                calls[key] = {"children": children, "neighbor_terms": neighbor_terms}
                call_meta[key] = {"parent_id": pid, "ref_to_cid": ref_to_cid}

        if not calls:
            continue

        # Seed the in-memory cache from the persistent (hash-keyed) store.
        cache: dict[str, dict] = {}
        for key, spec in calls.items():
            h = _group_hash(spec["children"], spec.get("neighbor_terms"), model)
            if h in persistent:
                cache[key] = persistent[h]

        results = await label_level_all(
            calls, model=model, corpus_topic=corpus_topic,
            child_level=child_level, parent_level=parent_name,
            concurrency=concurrency, cache=cache,
        )

        for key, entry in results.items():
            entries.append(entry)
            meta = call_meta[key]
            pid = meta["parent_id"]
            for ref, child in (entry.get("children") or {}).items():
                cid = meta["ref_to_cid"].get(ref)
                if cid is None:
                    continue
                out[ci][cid] = {
                    "label": child.get("label"),
                    "differs": child.get("differs"),
                    "parent_id": pid,
                    "parent_theme": entry.get("parent_theme"),
                    "parent_unifies": entry.get("parent_unifies"),
                }

    return out, entries
