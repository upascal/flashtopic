"""Clustering: pluggable flash-kmeans backends + the k-sweep."""

from __future__ import annotations

from flashtopic.cluster.backends import (
    BACKENDS,
    KMeansResult,
    available_backends,
    detect_backend,
    spherical_kmeans,
)
from flashtopic.cluster.sweep import SweepResult, sampled_silhouette, sweep_k
from flashtopic.cluster.levels import (
    LadderSelection,
    LevelsConfig,
    LevelsResult,
    NestingMetrics,
    StabilityReport,
    cluster_levels,
    level_stability,
    nesting_metrics,
    prune_small_clusters,
    select_ladder,
)

__all__ = [
    "BACKENDS",
    "KMeansResult",
    "available_backends",
    "detect_backend",
    "spherical_kmeans",
    "SweepResult",
    "sampled_silhouette",
    "sweep_k",
    "cluster_levels",
    "LevelsResult",
    "LevelsConfig",
    "NestingMetrics",
    "nesting_metrics",
    "level_stability",
    "StabilityReport",
    "prune_small_clusters",
    "select_ladder",
    "LadderSelection",
]
