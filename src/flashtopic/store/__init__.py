"""Optional persistence layer: keep chunks + embeddings + metadata in one keyed store.

The compute core is storeless; this is the *"way to save"*. :class:`FileStore` needs no
extra deps; :class:`DuckDBStore` (lazy — needs the ``[duckdb]`` extra) adds native vector
ops and a single-table, ``set_id``-keyed layout. Both satisfy :class:`VectorStore`.
"""

from __future__ import annotations

from flashtopic.store.base import Chunk, Row, VectorStore, chunks_from_result
from flashtopic.store.file_store import FileStore

__all__ = ["VectorStore", "Chunk", "Row", "chunks_from_result", "FileStore", "DuckDBStore"]


def __getattr__(name: str):
    # Lazy: only pull in duckdb when DuckDBStore is actually requested.
    if name == "DuckDBStore":
        from flashtopic.store.duckdb_store import DuckDBStore

        return DuckDBStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
