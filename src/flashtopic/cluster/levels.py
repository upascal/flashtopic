"""Nested multi-resolution clustering: a coherent ladder of K levels.

When no single K is obviously right, cluster at several resolutions and relate the levels
into a hierarchy (coarse → fine). Two ways to generate the levels:

* **flat** — independent spherical k-means at each requested K. Adjacent levels are
  stitched post-hoc (each fine cluster's dominant parent); nesting is *measured*, not
  guaranteed.
* **dendrogram** — one fine "atom" clustering, then an average-linkage tree over the atom
  centroids cut at several heights. Levels nest *by construction* (purity == 1).

This is a nested-cluster sweep idea, named ``levels`` here. Every scoring helper is pure and parameterized by
:class:`LevelsConfig` (defaults reproduce the original hand-tuned constants).
:func:`select_ladder` is the auto-selection pass built on them — rank nesting-coherent
cut sequences over one atom tree — for when you don't want to name the K's yourself;
:func:`cluster_levels` then materializes whichever ladder you settled on.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from flashtopic.cluster.backends import spherical_kmeans
from flashtopic.cluster.sweep import sampled_silhouette


@dataclass(frozen=True)
class LevelsConfig:
    """Weights + soft-band thresholds for the nesting/level scoring functions.

    Defaults reproduce the original hand-tuned values. Tweak via
    ``dataclasses.replace(cfg, field=value)``.
    """

    # nesting_metrics: how cleanly fine nests into coarse
    nesting_purity_w: float = 0.40
    nesting_homogeneity_w: float = 0.40
    nesting_parent_coverage_w: float = 0.10
    nesting_child_balance_w: float = 0.10

    # hierarchy_cut_rows: dendrogram-cut single-level score
    cut_silhouette_w: float = 0.40
    cut_height_gap_w: float = 0.30
    cut_atom_balance_w: float = 0.15
    cut_weight_balance_w: float = 0.15
    cut_singleton_penalty_w: float = 0.15
    cut_giant_penalty_w: float = 0.10
    cut_no_giant_band: tuple[float, float] = (0.70, 1.0)

    # hierarchy_pair_score: super × meta pair score
    pair_meta_cut_w: float = 0.28
    pair_super_cut_w: float = 0.22
    pair_meta_size_w: float = 0.20
    pair_super_child_count_w: float = 0.18
    pair_super_child_balance_w: float = 0.12
    pair_one_child_penalty_w: float = 0.10
    pair_target_atoms_per_meta: tuple[float, float] = (4.0, 9.0)
    pair_target_meta_per_super: tuple[float, float] = (2.0, 6.0)

    # quality_by_k: blended single-k quality used by sequence scoring
    quality_silhouette_w: float = 0.45
    quality_dist_w: float = 0.25
    quality_median_w: float = 0.20
    quality_max_w: float = 0.10
    quality_median_band_frac: tuple[float, float] = (0.003, 0.01)
    quality_max_ratio: float = 0.055
    quality_max_band: tuple[float, float] = (0.75, 1.0)

    # score_sequence: chain-of-k score
    sequence_mean_pair_w: float = 0.55
    sequence_min_pair_w: float = 0.20
    sequence_mean_quality_w: float = 0.25

    # pair filter for auto-selection
    min_pair_ratio: float = 1.5  # fine_k / coarse_k must exceed this to be scored


DEFAULT_LEVELS_CONFIG = LevelsConfig()


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def _as_inverse(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, inverse = np.unique(labels, return_inverse=True)
    return values, inverse.astype(np.int32)


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    nonzero = counts[counts > 0]
    if len(nonzero) <= 1:
        return 1.0
    return entropy_from_counts(nonzero) / float(np.log(len(nonzero)))


def robust01(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def target_band_score(value: float, low: float, high: float) -> float:
    """Soft 0..1 score where values inside [low, high] are ideal."""
    if value <= 0:
        return 0.0
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, min(1.0, value / low))
    return max(0.0, min(1.0, high / value))


# --------------------------------------------------------------------------- #
# Nesting between two flat clusterings
# --------------------------------------------------------------------------- #
def nested_contingency(
    coarse_labels: np.ndarray, fine_labels: np.ndarray
) -> np.ndarray:
    """Contingency table: rows are coarse clusters, columns are fine clusters."""
    if len(coarse_labels) != len(fine_labels):
        raise ValueError("coarse/fine labels must have the same length")
    _, coarse = _as_inverse(np.asarray(coarse_labels))
    _, fine = _as_inverse(np.asarray(fine_labels))
    out = np.zeros((int(coarse.max()) + 1, int(fine.max()) + 1), dtype=np.int64)
    np.add.at(out, (coarse, fine), 1)
    return out


@dataclass(frozen=True)
class NestingMetrics:
    coarse_k: int
    fine_k: int
    n: int
    purity: float
    homogeneity: float
    min_child_purity: float
    p10_child_purity: float
    parent_coverage: float
    child_balance: float
    score: float

    def as_dict(self) -> dict:
        return {
            "coarse_k": self.coarse_k,
            "fine_k": self.fine_k,
            "n": self.n,
            "purity": self.purity,
            "homogeneity": self.homogeneity,
            "min_child_purity": self.min_child_purity,
            "p10_child_purity": self.p10_child_purity,
            "parent_coverage": self.parent_coverage,
            "child_balance": self.child_balance,
            "score": self.score,
        }


def nesting_metrics(
    coarse_labels: np.ndarray,
    fine_labels: np.ndarray,
    *,
    coarse_k: int,
    fine_k: int,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> NestingMetrics:
    """How cleanly each fine cluster maps into one coarse parent."""
    table = nested_contingency(coarse_labels, fine_labels)
    n = int(table.sum())
    fine_sizes = table.sum(axis=0)
    coarse_sizes = table.sum(axis=1)
    child_max = table.max(axis=0)
    child_purity = child_max / np.maximum(fine_sizes, 1)
    purity = float(child_max.sum() / max(n, 1))

    h_coarse = entropy_from_counts(coarse_sizes)
    h_cond = 0.0
    for j, size in enumerate(fine_sizes):
        if size > 0:
            h_cond += float(size / max(n, 1)) * entropy_from_counts(table[:, j])
    homogeneity = 1.0 if h_coarse <= 0 else max(0.0, 1.0 - h_cond / h_coarse)

    parent_of_child = table.argmax(axis=0)
    parent_child_counts = np.bincount(parent_of_child, minlength=table.shape[0])
    parent_coverage = float((parent_child_counts > 0).mean())
    child_balance = normalized_entropy(parent_child_counts)
    score = (
        cfg.nesting_purity_w * purity
        + cfg.nesting_homogeneity_w * homogeneity
        + cfg.nesting_parent_coverage_w * parent_coverage
        + cfg.nesting_child_balance_w * child_balance
    )
    return NestingMetrics(
        coarse_k=coarse_k,
        fine_k=fine_k,
        n=n,
        purity=purity,
        homogeneity=homogeneity,
        min_child_purity=float(child_purity.min()) if len(child_purity) else 0.0,
        p10_child_purity=float(np.quantile(child_purity, 0.10))
        if len(child_purity)
        else 0.0,
        parent_coverage=parent_coverage,
        child_balance=child_balance,
        score=float(score),
    )


def nested_child_counts(
    coarse_labels: np.ndarray, fine_labels: np.ndarray
) -> np.ndarray:
    table = nested_contingency(coarse_labels, fine_labels)
    parent_of_child = table.argmax(axis=0)
    return np.bincount(parent_of_child, minlength=table.shape[0])


# --------------------------------------------------------------------------- #
# Agglomerative dendrogram cuts (nested by construction)
# --------------------------------------------------------------------------- #
def dendrogram_gap(Z: np.ndarray, k: int) -> float:
    """Height interval over which a maxclust cut keeps exactly k clusters."""
    n = Z.shape[0] + 1
    if k <= 1 or k >= n:
        return 0.0
    heights = Z[:, 2]
    merge_count = n - k
    prev_h = float(heights[merge_count - 1]) if merge_count > 0 else 0.0
    next_h = float(heights[merge_count]) if merge_count < len(heights) else prev_h
    return max(0.0, next_h - prev_h)


def cut_labels(Z: np.ndarray, k: int) -> np.ndarray:
    return fcluster(Z, t=k, criterion="maxclust").astype(np.int32)


def cut_size_summary(labels: np.ndarray, weights: np.ndarray | None = None) -> dict:
    _, inv = _as_inverse(labels)
    atom_counts = np.bincount(inv)
    if weights is None:
        weights = np.ones(len(labels), dtype=np.float64)
    weight_counts = np.bincount(inv, weights=np.asarray(weights, dtype=np.float64))
    total_weight = max(float(weight_counts.sum()), 1.0)
    return {
        "actual_k": int(len(atom_counts)),
        "atom_min": int(atom_counts.min()),
        "atom_median": float(np.median(atom_counts)),
        "atom_p90": float(np.quantile(atom_counts, 0.90)),
        "atom_max": int(atom_counts.max()),
        "singleton_frac": float((atom_counts == 1).mean()),
        "atom_balance": float(normalized_entropy(atom_counts)),
        "weight_median": float(np.median(weight_counts)),
        "weight_max_frac": float(weight_counts.max() / total_weight),
        "weight_balance": float(normalized_entropy(weight_counts)),
    }


def hierarchy_cut_rows(
    centroids: np.ndarray,
    Z: np.ndarray,
    cuts: list[int],
    *,
    weights: np.ndarray | None = None,
    seed: int = 0,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> list[dict]:
    """Per-cut single-level scores over an atom-centroid dendrogram."""
    centroids = centroids.astype(np.float32)
    centroids = centroids / np.clip(
        np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9, None
    )
    rows: list[dict] = []
    for k in cuts:
        labels = cut_labels(Z, k)
        summary = cut_size_summary(labels, weights)
        silhouette = sampled_silhouette(
            centroids, labels, sample=len(centroids), seed=seed
        )
        rows.append(
            {
                "cut_k": int(k),
                "silhouette": float(silhouette),
                "height_gap": float(dendrogram_gap(Z, k)),
                **summary,
            }
        )

    sil_norm = robust01([r["silhouette"] for r in rows])
    gap_norm = robust01([r["height_gap"] for r in rows])
    for i, row in enumerate(rows):
        no_giant_score = target_band_score(
            max(1e-9, 1.0 - row["weight_max_frac"]), *cfg.cut_no_giant_band
        )
        row["silhouette_norm"] = float(sil_norm[i])
        row["height_gap_norm"] = float(gap_norm[i])
        row["no_giant_score"] = float(no_giant_score)
        row["cut_score"] = float(
            cfg.cut_silhouette_w * sil_norm[i]
            + cfg.cut_height_gap_w * gap_norm[i]
            + cfg.cut_atom_balance_w * row["atom_balance"]
            + cfg.cut_weight_balance_w * row["weight_balance"]
            - cfg.cut_singleton_penalty_w * row["singleton_frac"]
            - cfg.cut_giant_penalty_w * (1.0 - no_giant_score)
        )
    return rows


def hierarchy_pair_score(
    super_row: dict,
    meta_row: dict,
    super_labels: np.ndarray,
    meta_labels: np.ndarray,
    *,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
    target_atoms_per_meta: tuple[float, float] | None = None,
    target_meta_per_super: tuple[float, float] | None = None,
) -> dict:
    atoms_band = target_atoms_per_meta or cfg.pair_target_atoms_per_meta
    metas_band = target_meta_per_super or cfg.pair_target_meta_per_super
    child_counts = nested_child_counts(super_labels, meta_labels)
    used = child_counts[child_counts > 0]
    child_median = float(np.median(used)) if len(used) else 0.0
    one_child_frac = float((used <= 1).mean()) if len(used) else 1.0
    child_balance = normalized_entropy(used)
    atoms_per_meta = float(meta_row["atom_median"])
    meta_size_score = target_band_score(atoms_per_meta, *atoms_band)
    child_count_score = target_band_score(child_median, *metas_band)
    score = (
        cfg.pair_meta_cut_w * float(meta_row["cut_score"])
        + cfg.pair_super_cut_w * float(super_row["cut_score"])
        + cfg.pair_meta_size_w * meta_size_score
        + cfg.pair_super_child_count_w * child_count_score
        + cfg.pair_super_child_balance_w * child_balance
        - cfg.pair_one_child_penalty_w * one_child_frac
    )
    return {
        "super_k": int(super_row["cut_k"]),
        "meta_k": int(meta_row["cut_k"]),
        "score": float(score),
        "super_cut_score": float(super_row["cut_score"]),
        "meta_cut_score": float(meta_row["cut_score"]),
        "meta_atom_median": atoms_per_meta,
        "meta_atom_max": int(meta_row["atom_max"]),
        "meta_singleton_frac": float(meta_row["singleton_frac"]),
        "super_meta_child_median": child_median,
        "super_meta_child_max": int(used.max()) if len(used) else 0,
        "super_one_child_frac": one_child_frac,
        "super_child_balance": float(child_balance),
        "meta_size_score": float(meta_size_score),
        "super_child_count_score": float(child_count_score),
    }


# --------------------------------------------------------------------------- #
# Auto-ladder selection helpers (used by a future flat-sequence selection pass)
# --------------------------------------------------------------------------- #
def quality_by_k(
    sweep_rows: dict[int, dict],
    ks: list[int],
    n_uniques: int,
    *,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> dict[int, float]:
    """Single-k quality blend used to weight a chain of k's.

    ``sweep_rows[k]`` must expose ``silhouette``, ``mean_cosine_distance``,
    ``cluster_median`` and ``cluster_max``. Bands are fractions of ``n_uniques`` so the
    score is corpus-size invariant.
    """
    sil = np.array([float(sweep_rows[k]["silhouette"]) for k in ks], dtype=np.float64)
    mean_dist = np.array(
        [float(sweep_rows[k]["mean_cosine_distance"]) for k in ks], dtype=np.float64
    )
    med = np.array(
        [float(sweep_rows[k]["cluster_median"]) for k in ks], dtype=np.float64
    )
    mx = np.array([float(sweep_rows[k]["cluster_max"]) for k in ks], dtype=np.float64)
    sil_n = robust01(sil)
    dist_n = robust01(-mean_dist)
    lo = cfg.quality_median_band_frac[0] * n_uniques
    hi = cfg.quality_median_band_frac[1] * n_uniques
    med_score = np.array([target_band_score(v, lo, hi) for v in med])
    max_ref = cfg.quality_max_ratio * n_uniques
    max_score = np.array(
        [
            target_band_score(
                max(1.0, max_ref / v) if v > 0 else 0.0, *cfg.quality_max_band
            )
            for v in mx
        ]
    )
    q = (
        cfg.quality_silhouette_w * sil_n
        + cfg.quality_dist_w * dist_n
        + cfg.quality_median_w * med_score
        + cfg.quality_max_w * max_score
    )
    return {int(k): float(v) for k, v in zip(ks, q)}


def score_sequence(
    pair_scores: list[float],
    qualities: list[float],
    *,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> dict[str, float]:
    """Score one k-chain; return the components alongside the total."""
    if not pair_scores or not qualities:
        return {"score": 0.0, "mean_pair": 0.0, "min_pair": 0.0, "mean_quality": 0.0}
    mp = float(np.mean(pair_scores))
    nm = float(min(pair_scores))
    mq = float(np.mean(qualities))
    return {
        "score": float(
            cfg.sequence_mean_pair_w * mp
            + cfg.sequence_min_pair_w * nm
            + cfg.sequence_mean_quality_w * mq
        ),
        "mean_pair": mp,
        "min_pair": nm,
        "mean_quality": mq,
    }


# --------------------------------------------------------------------------- #
# Auto-ladder selection: choose the k-chain instead of being handed one
# --------------------------------------------------------------------------- #
@dataclass
class LadderSelection:
    """The chosen coarse → fine ladder plus the evidence behind the choice."""

    ks: list[int]  # coarse → fine, e.g. [4, 9, 28]
    score: float  # score_sequence total for the winning chain
    components: dict  # its mean_pair / min_pair / mean_quality breakdown
    rows: list[dict]  # the hierarchy_cut_rows entry per chosen level
    n_atoms: int  # atoms the tree was actually built over (post-prune)
    candidates: list[int]  # every cut that was scored


def _log_candidates(lo: int, hi: int, count: int) -> list[int]:
    """``count`` roughly log-spaced integers in ``[lo, hi]`` (deduped, sorted)."""
    if hi < lo:
        return []
    if hi == lo:
        return [lo]
    vals = np.geomspace(lo, hi, num=max(2, count))
    return sorted({int(round(v)) for v in vals if lo <= round(v) <= hi})


def select_ladder(
    E: np.ndarray,
    *,
    n_levels: int = 3,
    atom_k: int | None = None,
    candidates: list[int] | None = None,
    backend: str = "auto",
    seed: int = 0,
    n_iter: int = 50,
    n_init: int = 4,
    min_size: int | None = None,
    max_mass_frac: float = 0.9,
    clusterer=None,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> LadderSelection:
    """Pick a nesting-coherent ladder of ``n_levels`` k's from the data itself.

    The auto-selection pass the scoring helpers above were written for. Cluster once at
    an atom resolution, build the average-linkage tree over the atom centroids, score
    many candidate cuts (:func:`hierarchy_cut_rows`) and every coarse→fine pair among
    them (:func:`hierarchy_pair_score`), then return the best strictly increasing chain
    (:func:`score_sequence`). Cuts are nearly free once the tree exists, so the whole
    search costs about one k-means — not one per candidate k.

    Each level's ``cut_score`` doubles as its quality term: it is the dendrogram path's
    native single-level score (silhouette + height gap + balance − singleton/giant).
    :func:`quality_by_k` is the flat-sweep equivalent and wants sweep rows, not cut rows.

    Pairs closer than ``cfg.min_pair_ratio`` are rejected so tiers stay distinct, and
    cuts whose largest cluster holds more than ``max_mass_frac`` of the corpus are
    dropped as undivided. A corpus too small to admit ``n_levels`` such tiers degrades
    to as many as fit — check ``len(result.ks)`` rather than assuming ``n_levels``.
    """
    E = _l2norm(E)
    n = E.shape[0]
    if n < 2:
        raise ValueError("need at least 2 points to select a ladder")
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")

    # Atoms only need to be fine enough to resolve the finest tier; the tree does the
    # rest. Same base construction as cluster_levels(strategy="dendrogram"), so the
    # selection matches what actually clustering at these cuts would produce.
    base_k = int(atom_k) if atom_k else min(256, max(16, n // 8))
    base_k = max(2, min(base_k, n - 1))
    atom_raw = _kmeans_labels(
        E, base_k, clusterer=clusterer, backend=backend, seed=seed,
        n_iter=n_iter, n_init=n_init,
    )
    _, atom_inv = _as_inverse(atom_raw)
    if min_size:
        atom_inv = prune_small_clusters(atom_inv, E, min_size)
    centroids = _centroids(E, atom_inv)
    n_atoms = int(centroids.shape[0])
    if n_atoms < 2:  # everything collapsed into one atom — nothing to ladder
        return LadderSelection(
            ks=[1], score=0.0, components=score_sequence([], []),
            rows=[], n_atoms=n_atoms, candidates=[],
        )

    Z = linkage(pdist(centroids, metric="cosine"), method="average")
    # Weight cuts by real corpus mass (points per atom), not atom counts — otherwise a
    # cut that isolates many tiny atoms looks as balanced as one that splits the bulk.
    weights = np.bincount(atom_inv, minlength=n_atoms).astype(np.float64)

    # Never consider a cut so fine that tiers are individual atoms — cfg's own band
    # wants at least that many atoms in the finest tier's clusters, and cuts near
    # n_atoms carry no structure the atom clustering didn't already decide.
    max_cut = max(2, int(n_atoms // cfg.pair_target_atoms_per_meta[0]))
    cand = (
        sorted({int(k) for k in candidates})
        if candidates
        else _log_candidates(2, max_cut, 16)
    )
    cand = [k for k in cand if 2 <= k <= max_cut]
    if not cand:
        cand = [max_cut]

    # Average linkage chains: near the top of the tree a cut peels off a few outlier
    # atoms and leaves one cluster holding ~all the corpus. That isn't a coarse tier,
    # it's an undivided corpus, and it would otherwise win on silhouette + height gap
    # (0.40 + 0.30 of cut_score) against a giant-cluster penalty of only 0.10. Drop
    # such cuts before scoring, so the normalization isn't anchored to them either.
    usable = [
        k for k in cand
        if cut_size_summary(cut_labels(Z, k), weights)["weight_max_frac"] <= max_mass_frac
    ]
    cand = usable or cand  # keep the degenerate set only if nothing else survives

    rows = hierarchy_cut_rows(centroids, Z, cand, weights=weights, seed=seed, cfg=cfg)
    row_by_k = {int(r["cut_k"]): r for r in rows}
    labels_by_k = {k: cut_labels(Z, k) for k in cand}
    quality = {k: float(row_by_k[k]["cut_score"]) for k in cand}

    # A tier must hold at least min_pair_ratio times the one above it, and cfg wants
    # each parent to have >= pair_target_meta_per_super[0] children — a ratio under
    # that means the average parent has fewer than two, so the tier nearly duplicates
    # the one below instead of summarizing it.
    min_ratio = max(cfg.min_pair_ratio, cfg.pair_target_meta_per_super[0])

    pair_cache: dict[tuple[int, int, int], float | None] = {}

    def _pair(coarse: int, fine: int, tiers: int) -> float | None:
        """Pair score, with the size band raised to the finer level's depth.

        ``hierarchy_pair_score`` bands the number of ATOMS per meta cluster, which is
        only the right unit when meta sits directly above the atoms. In a longer ladder
        a middle tier's children are the tier below it, not atoms — so a level ``tiers``
        above the atoms should hold about ``band ** tiers`` of them. Without this every
        tier is pulled to n_atoms/~6.5 and the ladder collapses toward one resolution.

        None means the pair is too close together to be distinct tiers.
        """
        key = (coarse, fine, tiers)
        if key not in pair_cache:
            if fine / coarse < min_ratio:
                pair_cache[key] = None
            else:
                lo, hi = cfg.pair_target_atoms_per_meta
                pair_cache[key] = float(
                    hierarchy_pair_score(
                        row_by_k[coarse], row_by_k[fine],
                        labels_by_k[coarse], labels_by_k[fine], cfg=cfg,
                        target_atoms_per_meta=(lo**tiers, hi**tiers),
                    )["score"]
                )
        return pair_cache[key]

    def _best_chain(m: int):
        """Highest-scoring strictly increasing chain of exactly ``m`` cuts, or None."""
        if m == 1:
            k = max(cand, key=lambda c: quality[c])
            return quality[k], [k], score_sequence([], [quality[k]], cfg=cfg)
        best = None
        for chain in combinations(cand, m):  # cand is sorted → chains increase
            # tier i+1 of the chain sits (m - 1 - i) levels above the atom base
            pairs = [
                _pair(a, b, m - 1 - i)
                for i, (a, b) in enumerate(zip(chain, chain[1:]))
            ]
            if any(p is None for p in pairs):
                continue  # some adjacent tier pair was too close together
            comp = score_sequence(pairs, [quality[k] for k in chain], cfg=cfg)
            if best is None or comp["score"] > best[0]:
                best = (comp["score"], list(chain), comp)
        return best

    chosen = None
    for m in range(min(n_levels, len(cand)), 0, -1):
        chosen = _best_chain(m)
        if chosen is not None:
            break

    score, ks, components = chosen
    return LadderSelection(
        ks=[int(k) for k in ks],
        score=float(score),
        components=components,
        rows=[row_by_k[k] for k in ks],
        n_atoms=n_atoms,
        candidates=list(cand),
    )


# --------------------------------------------------------------------------- #
# High-level entry point: materialize a ladder of nested levels
# --------------------------------------------------------------------------- #
@dataclass
class LevelsResult:
    """A coarse → fine ladder of clusterings plus how they nest.

    ``labels[i]`` are per-point cluster ids (contiguous, 0-based) at level ``i``;
    ``parents[i][c]`` is the level-``i`` parent of level-``(i+1)`` cluster ``c``;
    ``nesting[i]`` measures how cleanly level ``i+1`` nests into level ``i``.
    """

    strategy: str
    ks: list[int]  # actual cluster count per level, coarse → fine
    labels: list[np.ndarray]  # per-level (n,) point labels, coarse → fine
    parents: list[np.ndarray]  # parents[i]: fine cluster id → coarse cluster id
    nesting: list[NestingMetrics]  # between adjacent levels (len == len(ks) - 1)
    linkage: np.ndarray | None = None  # atom-centroid tree (dendrogram strategy only)


def _l2norm(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def _kmeans_labels(E, k, *, clusterer, backend, seed, n_iter, n_init) -> np.ndarray:
    if clusterer is not None:
        return np.asarray(clusterer.fit(E, k, seed=seed).labels)
    return spherical_kmeans(
        E, k, backend=backend, n_iter=n_iter, seed=seed, n_init=n_init
    ).labels


def _centroids(E: np.ndarray, labels: np.ndarray) -> np.ndarray:
    k = int(labels.max()) + 1
    C = np.zeros((k, E.shape[1]), dtype=np.float64)
    np.add.at(C, labels, E)
    counts = np.maximum(np.bincount(labels, minlength=k), 1).reshape(-1, 1)
    return _l2norm((C / counts).astype(np.float32))


def prune_small_clusters(
    labels: np.ndarray, X: np.ndarray, min_size: int
) -> np.ndarray:
    """Roll clusters smaller than ``min_size`` into their nearest surviving centroid.

    Kills singleton/trivia nodes so you only label clusters with real evidence. Returns
    contiguous 0-based labels; a no-op (still contiguous) if every cluster — or no
    cluster — meets ``min_size``.
    """
    X = _l2norm(X)
    _, inv = _as_inverse(np.asarray(labels))
    sizes = np.bincount(inv)
    keep = np.where(sizes >= int(min_size))[0]
    if len(keep) == 0 or len(keep) == len(sizes):
        return inv.astype(np.int64)
    keep_cents = _centroids(X, inv)[keep]  # (n_keep, d), L2-normalized
    out = inv.copy()
    small = ~np.isin(inv, keep)
    out[small] = keep[(X[small] @ keep_cents.T).argmax(axis=1)]
    return _as_inverse(out)[1].astype(np.int64)


def cluster_levels(
    E: np.ndarray,
    *,
    strategy: str = "flat",
    levels=None,
    cuts=None,
    atom_k: int | None = None,
    backend: str = "auto",
    seed: int = 0,
    n_iter: int = 50,
    n_init: int = 1,
    min_size: int | None = None,
    clusterer=None,
    config: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> LevelsResult:
    """Cluster ``E`` at several nested resolutions; return a coarse → fine ladder.

    ``strategy="flat"``: independent spherical k-means at each k in ``levels`` (nesting is
    measured, not guaranteed). ``strategy="dendrogram"``: cluster once at ``atom_k`` (or
    the finest ``cuts`` value), build an average-linkage tree over the atom centroids, and
    cut it at each k in ``cuts`` (levels nest by construction → purity 1.0).

    ``n_init`` restarts the k-means base and keeps the lowest-inertia fit — recommended
    (≥4) for the dendrogram atom step, whose local minima would otherwise shift the whole
    tree on a re-run. ``min_size`` drops clusters with fewer than that many members,
    rolling them into the nearest surviving centroid, so trivia/singleton nodes never
    reach labeling; for ``dendrogram`` it is applied to the atom base, which keeps every
    cut ≥ ``min_size`` *and* preserves perfect nesting. Pass a custom ``clusterer`` (the
    :class:`~flashtopic.protocols.Clusterer` protocol) to override the default k-means.
    """
    E = _l2norm(E)
    Z = None
    if strategy == "flat":
        req = sorted({int(k) for k in (levels or [])})
        if not req:
            raise ValueError("strategy='flat' needs levels=[k1, k2, ...]")
        labels = [
            _kmeans_labels(
                E,
                k,
                clusterer=clusterer,
                backend=backend,
                seed=seed,
                n_iter=n_iter,
                n_init=n_init,
            )
            for k in req
        ]
        if min_size:
            labels = [prune_small_clusters(lab, E, min_size) for lab in labels]
    elif strategy == "dendrogram":
        cut_ks = sorted(
            {int(k) for k in (cuts if cuts is not None else (levels or []))}
        )
        if not cut_ks:
            raise ValueError("strategy='dendrogram' needs cuts=[k1, k2, ...]")
        atom_k = int(atom_k) if atom_k else cut_ks[-1]
        if atom_k < cut_ks[-1]:
            raise ValueError(
                f"atom_k ({atom_k}) must be >= the finest cut ({cut_ks[-1]})"
            )
        atom_raw = _kmeans_labels(
            E,
            atom_k,
            clusterer=clusterer,
            backend=backend,
            seed=seed,
            n_iter=n_iter,
            n_init=n_init,
        )
        _, atom_inv = _as_inverse(atom_raw)  # contiguous atom ids
        if min_size:  # prune the base → all cuts stay ≥ min_size and nest perfectly
            atom_inv = prune_small_clusters(atom_inv, E, min_size)
        centroids = _centroids(E, atom_inv)
        Z = linkage(pdist(centroids, metric="cosine"), method="average")
        a = centroids.shape[0]
        labels = [cut_labels(Z, min(k, a))[atom_inv] for k in cut_ks]
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use 'flat' or 'dendrogram'")

    # contiguous 0-based labels per level so parents[i] indexes directly by fine label
    labels = [_as_inverse(lab)[1].astype(np.int64) for lab in labels]
    ks = [int(lab.max()) + 1 for lab in labels]

    parents: list[np.ndarray] = []
    nesting: list[NestingMetrics] = []
    for i in range(len(labels) - 1):
        coarse, fine = labels[i], labels[i + 1]
        parents.append(nested_contingency(coarse, fine).argmax(axis=0))
        nesting.append(
            nesting_metrics(coarse, fine, coarse_k=ks[i], fine_k=ks[i + 1], cfg=config)
        )
    return LevelsResult(
        strategy=strategy,
        ks=ks,
        labels=labels,
        parents=parents,
        nesting=nesting,
        linkage=Z,
    )


# --------------------------------------------------------------------------- #
# Stability gate: is a resolution reproducible enough to commit/label?
# --------------------------------------------------------------------------- #
@dataclass
class StabilityReport:
    """Per-level label agreement (adjusted Rand index) across seeds, coarse → fine.

    Low fine-level ARI means the resolution shifts run-to-run — clustering there is not
    reproducible, so committing to labels/prevalence on it is premature. ``ks`` are the
    cluster counts from the first seed's run.
    """

    seeds: list[int]
    ks: list[int]
    mean_ari: list[float]
    min_ari: list[float]

    def as_dict(self) -> dict:
        return {
            "seeds": self.seeds,
            "ks": self.ks,
            "mean_ari": self.mean_ari,
            "min_ari": self.min_ari,
        }


def level_stability(E: np.ndarray, *, seeds=(0, 1, 2, 3), **kwargs) -> StabilityReport:
    """Re-run :func:`cluster_levels` at several seeds and score per-level reproducibility.

    Returns mean and min pairwise ARI per level. Use it as the gate *before* spending on
    LLM labels: if fine-level ARI is low, raise ``n_init``, prune (``min_size``), or back
    off to a coarser level. ``kwargs`` are forwarded to :func:`cluster_levels` (``strategy``,
    ``levels`` / ``cuts``, ``atom_k``, ``n_init``, ``min_size``, ``backend``); do not pass
    ``seed``.
    """
    from sklearn.metrics import adjusted_rand_score

    seeds = list(seeds)
    if len(seeds) < 2:
        raise ValueError("need >= 2 seeds to measure stability")
    runs = [cluster_levels(E, seed=s, **kwargs) for s in seeds]
    n_levels = min(len(r.labels) for r in runs)
    mean_ari, min_ari, ks = [], [], []
    for li in range(n_levels):
        pair = [
            adjusted_rand_score(runs[a].labels[li], runs[b].labels[li])
            for a in range(len(seeds))
            for b in range(a + 1, len(seeds))
        ]
        mean_ari.append(float(np.mean(pair)))
        min_ari.append(float(np.min(pair)))
        ks.append(int(runs[0].ks[li]))
    return StabilityReport(seeds=seeds, ks=ks, mean_ari=mean_ari, min_ari=min_ari)
