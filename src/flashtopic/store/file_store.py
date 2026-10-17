"""FileStore — the zero-dependency :class:`VectorStore` over the current ``.npy``+JSONL layout.

A "set" is a directory ``base_dir/<set_id>/`` holding ``embeddings.npy`` (rows aligned to
membership order) and ``corpus/body_membership.jsonl`` — exactly the paths a
:class:`~flashtopic.config.Workspace` already uses, so a run written by the pipeline is
readable as a set and vice versa. It proves the Protocol with no extra deps and keeps
``RunContext.from_store`` working even when ``duckdb`` isn't installed.

It is deliberately simple: ``upsert`` rewrites the whole set (fine for write-once runs;
use :class:`~flashtopic.store.DuckDBStore` for incremental, keyed updates at scale), and
metadata/vector filtering is brute force in NumPy. ``where`` / ``meta_predicate`` are
``Callable[[dict], bool]`` over each row dict (keys: ``chunk_id, text, metadata,
doc_idxs``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from flashtopic.corpus import UniqueIndex
from flashtopic.store.base import Chunk, Row


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-9)


class FileStore:
    """``VectorStore`` backed by per-set ``embeddings.npy`` + ``body_membership.jsonl``."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    @classmethod
    def from_workspace(cls, ws) -> "FileStore":
        return cls(ws.base_dir)

    # ---- paths ------------------------------------------------------------ #
    def _set_dir(self, set_id: str) -> Path:
        return self.base_dir / set_id

    def _emb_path(self, set_id: str) -> Path:
        return self._set_dir(set_id) / "embeddings.npy"

    def _membership_path(self, set_id: str) -> Path:
        return self._set_dir(set_id) / "corpus" / "body_membership.jsonl"

    def _meta_path(self, set_id: str) -> Path:
        return self._set_dir(set_id) / "set_meta.json"

    # ---- set management --------------------------------------------------- #
    def create_set(self, set_id: str, *, dim: int, model: str = "",
                   metadata: dict | None = None) -> None:
        self._set_dir(set_id).mkdir(parents=True, exist_ok=True)
        self._meta_path(set_id).write_text(
            json.dumps({"dim": int(dim), "model": model, "metadata": metadata or {}})
        )

    def list_sets(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            p.name for p in self.base_dir.iterdir()
            if p.is_dir() and ((p / "embeddings.npy").exists() or (p / "set_meta.json").exists())
        )

    def drop_set(self, set_id: str) -> None:
        shutil.rmtree(self._set_dir(set_id), ignore_errors=True)

    # ---- load helpers ----------------------------------------------------- #
    def _load(self, set_id: str) -> tuple[list[dict], np.ndarray]:
        """Return (row dicts in stored order, aligned (n, d) embedding matrix)."""
        mpath = self._membership_path(set_id)
        epath = self._emb_path(set_id)
        if not mpath.exists() or not epath.exists():
            raise FileNotFoundError(f"set {set_id!r} not found under {self.base_dir}")
        rows: list[dict] = []
        for line in mpath.read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "chunk_id": str(r.get("chunk_id", r.get("unique_id"))),
                "text": r["text"],
                "doc_idxs": list(r.get("doc_idxs", [])),
                "sentence_idxs": list(r.get("sentence_idxs", [])),
                "metadata": r.get("metadata", {}),
            })
        E = np.load(epath).astype(np.float32)
        if E.shape[0] != len(rows):
            raise ValueError(f"set {set_id!r}: {E.shape[0]} embeddings vs {len(rows)} membership rows")
        return rows, E

    def _write(self, set_id: str, rows: list[dict], E: np.ndarray) -> None:
        self._set_dir(set_id).mkdir(parents=True, exist_ok=True)
        self._membership_path(set_id).parent.mkdir(parents=True, exist_ok=True)
        np.save(self._emb_path(set_id), np.asarray(E, dtype=np.float32))
        with self._membership_path(set_id).open("w") as f:
            for i, r in enumerate(rows):
                rec = {"unique_id": i, "chunk_id": r["chunk_id"], "text": r["text"],
                       "count": len(r["sentence_idxs"]) or len(r["doc_idxs"]),
                       "sentence_idxs": r["sentence_idxs"], "doc_idxs": r["doc_idxs"]}
                if r["metadata"]:
                    rec["metadata"] = r["metadata"]
                f.write(json.dumps(rec) + "\n")

    # ---- writes ----------------------------------------------------------- #
    def upsert(self, set_id: str, chunks: Sequence[Chunk]) -> None:
        try:
            rows, E = self._load(set_id)
        except FileNotFoundError:
            rows, E = [], np.empty((0, 0), dtype=np.float32)
        by_id = {r["chunk_id"]: i for i, r in enumerate(rows)}
        emb_list = [E[i] for i in range(len(rows))]
        for ch in chunks:
            rec = {"chunk_id": ch.chunk_id, "text": ch.text, "doc_idxs": list(ch.doc_idxs),
                   "sentence_idxs": list(ch.sentence_idxs), "metadata": dict(ch.metadata)}
            vec = np.asarray(ch.embedding, dtype=np.float32)
            if ch.chunk_id in by_id:
                j = by_id[ch.chunk_id]
                rows[j] = rec
                emb_list[j] = vec
            else:
                by_id[ch.chunk_id] = len(rows)
                rows.append(rec)
                emb_list.append(vec)
        self._write(set_id, rows, np.stack(emb_list) if emb_list else np.empty((0, 0), np.float32))

    # ---- reads ------------------------------------------------------------ #
    def _filter_idx(self, rows: list[dict], where: Callable[[dict], bool] | None,
                    ids: Sequence[str] | None) -> list[int]:
        id_set = set(ids) if ids is not None else None
        out = []
        for i, r in enumerate(rows):
            if id_set is not None and r["chunk_id"] not in id_set:
                continue
            if where is not None and not where(r):
                continue
            out.append(i)
        return out

    def get_matrix(self, set_id: str, *, where: Any = None,
                   ids: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
        rows, E = self._load(set_id)
        idx = self._filter_idx(rows, where, ids)
        M = np.ascontiguousarray(E[idx], dtype=np.float32) if idx else np.empty((0, E.shape[1] if E.ndim == 2 else 0), np.float32)
        return M, [rows[i]["chunk_id"] for i in idx]

    def query(self, set_id: str, *, vector: np.ndarray | None = None,
              meta_predicate: Any = None, top_k: int | None = None,
              min_similarity: float | None = None) -> list[Row]:
        rows, E = self._load(set_id)
        idx = self._filter_idx(rows, meta_predicate, None)
        if not idx:
            return []
        if vector is not None:
            q = _l2(vector)
            sims = (E[idx] @ q).tolist()
        else:
            sims = [None] * len(idx)
        out = [Row(chunk_id=rows[i]["chunk_id"], text=rows[i]["text"], score=s,
                   metadata=rows[i]["metadata"], doc_idxs=rows[i]["doc_idxs"])
               for i, s in zip(idx, sims)]
        if min_similarity is not None:
            out = [r for r in out if r.score is not None and r.score >= min_similarity]
        if vector is not None:
            out.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            out = out[:top_k]
        return out

    def get_unique_index(self, set_id: str, *, where: Any = None
                         ) -> tuple[UniqueIndex, np.ndarray]:
        rows, E = self._load(set_id)
        idx = self._filter_idx(rows, where, None)
        ui = UniqueIndex(
            texts=[rows[i]["text"] for i in idx],
            sentence_idxs=[rows[i]["sentence_idxs"] for i in idx],
            doc_idxs=[rows[i]["doc_idxs"] for i in idx],
        )
        M = np.ascontiguousarray(E[idx], dtype=np.float32) if idx else np.empty((0, E.shape[1] if E.ndim == 2 else 0), np.float32)
        return ui, M

    def doc_texts(self, set_id: str) -> dict[int, str]:
        """Reconstruct each document's text by joining its atoms (best-effort, body only)."""
        rows, _ = self._load(set_id)
        by_doc: dict[int, list[str]] = {}
        for r in rows:
            for di in r["doc_idxs"]:
                by_doc.setdefault(int(di), []).append(r["text"])
        return {d: "\n".join(t) for d, t in by_doc.items()}
