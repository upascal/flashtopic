"""VectorStore — the optional persistence layer (chunks + embeddings + metadata together).

The compute core (:func:`flashtopic.cluster_corpus`) is storeless: arrays in, a
``ClusterResult`` out. This Protocol is the optional *"way to save"* — a place that keeps
each atom's text, embedding, membership and arbitrary metadata in **one keyed store**, so
that managing many embedding *sets* (e.g. one per root query) is a ``set_id`` column
rather than a sprawl of ``.npy`` files, and a ``persons = X AND cosine > 0.7`` filter is a
single query instead of a hand-rolled join between a ``.npy`` and a metadata file.

Two backends ship: :class:`flashtopic.store.FileStore` (zero extra deps, the current
``.npy``+JSONL layout) and :class:`flashtopic.store.DuckDBStore` (columnar, native vector
ops, web-friendly). sqlite-vec / pgvector are conforming alternates a host project can
add. The core never imports this module — it stays optional.

``where`` / ``meta_predicate`` are **backend-specific** filters: a SQL fragment for
``DuckDBStore`` (e.g. ``"json_extract_string(metadata,'$.persons') LIKE '%X%'"``), or a
``Callable[[dict], bool]`` over a chunk's metadata for ``FileStore``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from flashtopic.corpus import UniqueIndex


@dataclass
class Chunk:
    """One atom (unique sentence) to store: its key, text, vector, membership, metadata."""

    chunk_id: str
    text: str
    embedding: np.ndarray  # (d,) float32 (stored L2-normalized)
    doc_idxs: list[int] = field(default_factory=list)  # documents containing this atom
    sentence_idxs: list[int] = field(default_factory=list)  # optional provenance
    metadata: dict = field(default_factory=dict)  # arbitrary per-chunk metadata
    doc_id: str | None = None


@dataclass
class Row:
    """A query/read result: the chunk plus an optional similarity ``score``."""

    chunk_id: str
    text: str
    score: float | None = None
    metadata: dict = field(default_factory=dict)
    doc_idxs: list[int] = field(default_factory=list)


@runtime_checkable
class VectorStore(Protocol):
    """Keyed store of chunks + embeddings + metadata, partitioned into named sets."""

    # ---- set management --------------------------------------------------- #
    def create_set(self, set_id: str, *, dim: int, model: str = "",
                   metadata: dict | None = None) -> None: ...

    def list_sets(self) -> list[str]: ...

    def drop_set(self, set_id: str) -> None: ...

    # ---- writes ----------------------------------------------------------- #
    def upsert(self, set_id: str, chunks: Sequence[Chunk]) -> None: ...

    # ---- reads ------------------------------------------------------------ #
    def get_matrix(self, set_id: str, *, where: Any = None,
                   ids: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Materialize a contiguous ``(n, d)`` float32 matrix + aligned chunk_ids."""
        ...

    def query(self, set_id: str, *, vector: np.ndarray | None = None,
              meta_predicate: Any = None, top_k: int | None = None,
              min_similarity: float | None = None) -> list[Row]:
        """Filter by metadata and/or cosine similarity to ``vector`` in one pass."""
        ...

    def get_unique_index(self, set_id: str, *, where: Any = None
                         ) -> tuple[UniqueIndex, np.ndarray]:
        """Rebuild a :class:`UniqueIndex` + embedding matrix to feed clustering/prevalence."""
        ...

    def doc_texts(self, set_id: str) -> dict[int, str]:
        """Reconstruct ``doc_idx → text`` (best-effort, body atoms) for the judge."""
        ...


def chunks_from_result(result) -> list[Chunk]:
    """Convert a :class:`flashtopic.pipeline.ClusterResult` into storable chunks.

    The chunk key is the unique-sentence id (``str(uid)``); row order is preserved so the
    embedding matrix stays aligned with membership on read-back.
    """
    ui = result.unique_index
    E = np.asarray(result.embeddings, dtype=np.float32)
    return [
        Chunk(
            chunk_id=str(uid),
            text=ui.texts[uid],
            embedding=E[uid],
            doc_idxs=list(ui.doc_idxs[uid]),
            sentence_idxs=list(ui.sentence_idxs[uid]),
        )
        for uid in range(len(ui.texts))
    ]
