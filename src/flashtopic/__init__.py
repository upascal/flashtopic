"""flashtopic — fast GPU topic modeling + debiased prevalence scoring.

Pipeline: ingest → sentence parse → NER → flash k-means sweep → distinctive-term
labels → contrastive LLM labeling → debiased prevalence estimation.

Public API is re-exported here as the stages land; ``Workspace`` is the config object
every stage takes.
"""

from __future__ import annotations

__version__ = "0.2.0"

from flashtopic.config import Workspace
from flashtopic.schema import Doc
from flashtopic.ingest import load_docs, to_docs
from flashtopic.corpus import (
    SentenceRecord,
    UniqueIndex,
    build_sentence_corpus,
    build_unique_index,
)
from flashtopic.terms import distinctive_terms
from flashtopic.sampling import coverage_exemplars
from flashtopic.embed import get_encoder, embed_corpus, embed_texts, sniff_backend
from flashtopic.cluster import spherical_kmeans, sweep_k, detect_backend, available_backends
from flashtopic.cluster import cluster_levels, LevelsResult, LevelsConfig
from flashtopic.cluster import level_stability, StabilityReport, prune_small_clusters
from flashtopic.pipeline import cluster_corpus, ClusterResult, run_pipeline, PipelineResult
from flashtopic.label_hierarchy import label_levels
from flashtopic.protocols import Clusterer, Embedder, Encoder, Labeler, SphericalKMeans, TermScorer
from flashtopic.store import Chunk, FileStore, Row, VectorStore
from flashtopic.llm import configure_llm

__all__ = [
    "Workspace",
    "Doc",
    "load_docs",
    "to_docs",
    "SentenceRecord",
    "UniqueIndex",
    "build_sentence_corpus",
    "build_unique_index",
    "distinctive_terms",
    "coverage_exemplars",
    "get_encoder",
    "sniff_backend",
    "embed_corpus",
    "embed_texts",
    "spherical_kmeans",
    "sweep_k",
    "detect_backend",
    "available_backends",
    "cluster_levels",
    "LevelsResult",
    "LevelsConfig",
    "level_stability",
    "StabilityReport",
    "prune_small_clusters",
    "cluster_corpus",
    "ClusterResult",
    "run_pipeline",
    "PipelineResult",
    "label_levels",
    "Clusterer",
    "Embedder",
    "Encoder",
    "Labeler",
    "SphericalKMeans",
    "TermScorer",
    "VectorStore",
    "Chunk",
    "Row",
    "FileStore",
    "DuckDBStore",
    "configure_llm",
    "__version__",
]


def __getattr__(name: str):
    # DuckDBStore is lazy so the base install need not have the [duckdb] extra.
    if name == "DuckDBStore":
        from flashtopic.store import DuckDBStore

        return DuckDBStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
