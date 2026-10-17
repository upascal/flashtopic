"""Swappable component Protocols — flashtopic's BERTopic-style extension points.

flashtopic is a general-purpose clustering pipeline: every stage is a duck-typed
component you can replace without subclassing. These Protocols name the contracts so
consumers in other projects can write conforming implementations — a custom embedder, a
different clusterer, an alternative term scorer or labeler — and pass them into
:func:`flashtopic.cluster_corpus`. There is **no BERTopic dependency**; we mirror its
conventions (swappable pieces, precomputed-``embeddings=`` in) without importing it.

Defaults stay exactly as the pipeline already uses them: :class:`SphericalKMeans` over
the flash-kmeans backends, ``terms.distinctive_terms`` for terms, the OpenRouter LLM
gateway for labels. The Protocols are about *opening* those slots, not changing them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from flashtopic.embed import Encoder
from flashtopic.terms import TermScorer

# An Embedder is exactly the existing ``Encoder`` protocol — alias it so consumers can
# import either name from one place. Returns ``(N, d)`` float32, L2-normalized.
Embedder = Encoder


@runtime_checkable
class Clusterer(Protocol):
    """Partition ``X`` (n, d, L2-normalized) into ``k`` clusters.

    ``fit`` returns any object exposing a ``.labels`` ``(n,)`` int array — e.g.
    :class:`flashtopic.cluster.KMeansResult`. The default used by ``cluster_corpus`` when
    no clusterer is passed is :class:`SphericalKMeans`.
    """

    def fit(self, X: np.ndarray, k: int, *, seed: int = 0) -> Any: ...


class SphericalKMeans:
    """Default :class:`Clusterer`: spherical (cosine) k-means via the flash backends."""

    def __init__(self, backend: str = "auto"):
        self.backend = backend

    def fit(self, X: np.ndarray, k: int, *, seed: int = 0):
        from flashtopic.cluster import spherical_kmeans

        return spherical_kmeans(X, k, backend=self.backend, seed=seed)


@runtime_checkable
class Labeler(Protocol):
    """Turn per-cluster evidence into labels (async; e.g. an LLM gateway).

    ``evidence`` maps cluster id → :class:`flashtopic.label.AtomEvidence`; the return maps
    cluster id → a dict carrying at least ``specific_topic``. The default is
    ``flashtopic.label.label_atoms_all``.
    """

    async def label(self, evidence: dict[int, Any], *, model: str,
                    corpus_topic: str = "") -> dict[int, dict]: ...


__all__ = ["Embedder", "Encoder", "Clusterer", "SphericalKMeans", "TermScorer", "Labeler"]
