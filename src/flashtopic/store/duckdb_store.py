"""DuckDBStore — columnar :class:`VectorStore` with native vector ops (the ``[duckdb]`` extra).

One ``chunks`` table keyed by ``(set_id, chunk_id)`` holds text + embedding + membership +
metadata for every atom of every set — so managing many embedding sets is a ``set_id``
column, not a sprawl of ``.npy`` files, and a metadata + similarity filter is one SQL
query (``... WHERE persons = 'X' AND list_cosine_similarity(embedding, q) > 0.7``) instead
of a hand-rolled join between a ``.npy`` and a sidecar.

Embeddings are stored as ``FLOAT[]`` and scored with DuckDB's built-in
``list_cosine_similarity`` (brute force — exact, and fast enough for the per-set scale this
library targets, ≤~400k×512; it also keeps the schema DuckDB-WASM compatible for a web
browser). An HNSW index via the ``vss`` extension is a future option for cross-set
retrieval; it would require fixed-size ``FLOAT[d]`` columns and is intentionally not wired
in here.

``where`` / ``meta_predicate`` are **trusted SQL fragments** (this is a SQL store): e.g.
``"json_extract_string(metadata, '$.persons') = 'X'"`` or, if you promoted ``persons`` via
``metadata_columns=["persons"]``, just ``"persons = 'X'"``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

import numpy as np

from flashtopic.corpus import UniqueIndex
from flashtopic.store.base import Chunk, Row

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BASE_COLS = ("set_id", "chunk_id", "ordinal", "doc_id", "text",
              "embedding", "doc_idxs", "sentence_idxs", "metadata")


def _promote(meta: dict, col: str):
    v = meta.get(col)
    if v is None or isinstance(v, str):
        return v
    return json.dumps(v)


class DuckDBStore:
    """``VectorStore`` over a DuckDB database (``:memory:`` or a ``.duckdb`` file)."""

    def __init__(self, path: str = ":memory:", *, metadata_columns: Sequence[str] | None = None):
        import duckdb

        for c in metadata_columns or []:
            if not _IDENT.match(c):
                raise ValueError(f"metadata column {c!r} is not a valid identifier")
        self.metadata_columns = list(metadata_columns or [])
        self.con = duckdb.connect(path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS embedding_sets ("
            "set_id VARCHAR PRIMARY KEY, dim INTEGER, model VARCHAR, metadata JSON)"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "set_id VARCHAR NOT NULL, chunk_id VARCHAR NOT NULL, ordinal INTEGER, "
            "doc_id VARCHAR, text VARCHAR, embedding FLOAT[], "
            "doc_idxs INTEGER[], sentence_idxs INTEGER[], metadata JSON, "
            "PRIMARY KEY (set_id, chunk_id))"
        )
        existing = {r[0] for r in self.con.execute("PRAGMA table_info('chunks')").fetchall()}
        for c in self.metadata_columns:
            if c not in existing:
                self.con.execute(f"ALTER TABLE chunks ADD COLUMN {c} VARCHAR")

    # ---- set management --------------------------------------------------- #
    def create_set(self, set_id: str, *, dim: int, model: str = "",
                   metadata: dict | None = None) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO embedding_sets (set_id, dim, model, metadata) VALUES (?,?,?,?)",
            [set_id, int(dim), model, json.dumps(metadata or {})],
        )

    def list_sets(self) -> list[str]:
        return [r[0] for r in self.con.execute(
            "SELECT set_id FROM embedding_sets ORDER BY set_id").fetchall()]

    def drop_set(self, set_id: str) -> None:
        self.con.execute("DELETE FROM chunks WHERE set_id = ?", [set_id])
        self.con.execute("DELETE FROM embedding_sets WHERE set_id = ?", [set_id])

    # ---- writes ----------------------------------------------------------- #
    def upsert(self, set_id: str, chunks: Sequence[Chunk]) -> None:
        existing = dict(self.con.execute(
            "SELECT chunk_id, ordinal FROM chunks WHERE set_id = ?", [set_id]).fetchall())
        next_ord = (max(existing.values()) + 1) if existing else 0
        cols = list(_BASE_COLS) + self.metadata_columns
        ph = ",".join(["?"] * len(cols))
        rows = []
        for ch in chunks:
            if ch.chunk_id in existing:
                o = existing[ch.chunk_id]
            else:
                o = next_ord
                next_ord += 1
                existing[ch.chunk_id] = o
            rows.append([
                set_id, ch.chunk_id, o, ch.doc_id, ch.text,
                np.asarray(ch.embedding, dtype=np.float32).tolist(),
                [int(d) for d in ch.doc_idxs], [int(s) for s in ch.sentence_idxs],
                json.dumps(ch.metadata),
                *[_promote(ch.metadata, c) for c in self.metadata_columns],
            ])
        if rows:
            self.con.executemany(
                f"INSERT OR REPLACE INTO chunks ({', '.join(cols)}) VALUES ({ph})", rows)

    # ---- reads ------------------------------------------------------------ #
    def _where(self, set_id: str, extra: str | None, ids: Sequence[str] | None):
        clauses = ["set_id = ?"]
        params: list[Any] = [set_id]
        if extra:
            clauses.append(f"({extra})")
        if ids is not None:
            ids = list(ids)
            clauses.append(f"chunk_id IN ({','.join(['?'] * len(ids))})")
            params.extend(ids)
        return " AND ".join(clauses), params

    def _matrix(self, embs) -> np.ndarray:
        return np.stack([np.asarray(e, dtype=np.float32) for e in embs]) if len(embs) \
            else np.empty((0, 0), dtype=np.float32)

    def get_matrix(self, set_id: str, *, where: Any = None,
                   ids: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
        cond, params = self._where(set_id, where, ids)
        res = self.con.execute(
            f"SELECT chunk_id, embedding FROM chunks WHERE {cond} ORDER BY ordinal", params
        ).fetchall()
        M = self._matrix([e for _, e in res])
        return np.ascontiguousarray(M), [c for c, _ in res]

    def query(self, set_id: str, *, vector: np.ndarray | None = None,
              meta_predicate: Any = None, top_k: int | None = None,
              min_similarity: float | None = None) -> list[Row]:
        cond, params = self._where(set_id, meta_predicate, None)
        if vector is not None:
            q = np.asarray(vector, dtype=np.float32)
            q = (q / max(float(np.linalg.norm(q)), 1e-9)).tolist()
            inner = (f"SELECT chunk_id, text, metadata, doc_idxs, "
                     f"list_cosine_similarity(embedding, ?::FLOAT[]) AS score "
                     f"FROM chunks WHERE {cond}")
            sql = f"SELECT * FROM ({inner})"
            iparams = [q] + params
            if min_similarity is not None:
                sql += " WHERE score >= ?"
                iparams.append(float(min_similarity))
            sql += " ORDER BY score DESC"
        else:
            sql = (f"SELECT chunk_id, text, metadata, doc_idxs, NULL AS score "
                   f"FROM chunks WHERE {cond}")
            iparams = params
        if top_k is not None:
            sql += f" LIMIT {int(top_k)}"
        out = []
        for cid, text, meta, doc_idxs, score in self.con.execute(sql, iparams).fetchall():
            out.append(Row(chunk_id=cid, text=text,
                           score=None if score is None else float(score),
                           metadata=json.loads(meta) if meta else {},
                           doc_idxs=list(doc_idxs or [])))
        return out

    def get_unique_index(self, set_id: str, *, where: Any = None
                         ) -> tuple[UniqueIndex, np.ndarray]:
        cond, params = self._where(set_id, where, None)
        res = self.con.execute(
            f"SELECT text, embedding, doc_idxs, sentence_idxs FROM chunks "
            f"WHERE {cond} ORDER BY ordinal", params
        ).fetchall()
        ui = UniqueIndex(
            texts=[r[0] for r in res],
            sentence_idxs=[list(r[3] or []) for r in res],
            doc_idxs=[list(r[2] or []) for r in res],
        )
        return ui, np.ascontiguousarray(self._matrix([r[1] for r in res]))

    def doc_texts(self, set_id: str) -> dict[int, str]:
        """Reconstruct each document's text by joining its atoms (best-effort, body only)."""
        res = self.con.execute(
            "SELECT text, doc_idxs FROM chunks WHERE set_id = ? ORDER BY ordinal", [set_id]
        ).fetchall()
        by_doc: dict[int, list[str]] = {}
        for text, doc_idxs in res:
            for di in (doc_idxs or []):
                by_doc.setdefault(int(di), []).append(text)
        return {d: "\n".join(t) for d, t in by_doc.items()}

    def close(self) -> None:
        self.con.close()
