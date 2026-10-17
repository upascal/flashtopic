"""RunContext — the corpus side the prevalence estimators read from.

Generalized from an earlier pilot's ``pilot_data``: that one had to *reproduce* the
clustering to recover membership, because the deduped topic specs didn't store it.
flashtopic's clustering stores per-unique cluster labels and per-unique → document
membership directly, so the context is built straight from pipeline outputs (in
memory) or from a run's on-disk artifacts — no re-clustering.

A ``RunContext`` provides, for a run: the unique-sentence embeddings ``E``, the unique
texts, the unique → document membership (and its inverse ``doc_uids``), the document
count, and each document's full text (for the judge).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


@dataclass
class RunContext:
    E: np.ndarray  # (N_uniq, d) L2-normalized unique-sentence embeddings
    texts: list[str]  # unique texts (index = unique_id)
    uid_docs: list[list[int]]  # unique_id → sorted distinct document indices
    n_docs: int  # number of documents (prevalence denominator)
    doc_texts: dict[int, str]  # document index → full text (for the judge)
    doc_uids: dict[int, list[int]] = field(init=False)  # document index → unique_ids present

    def __post_init__(self):
        self.E = _l2(self.E)
        d: dict[int, list[int]] = {}
        for uid, docs in enumerate(self.uid_docs):
            for di in docs:
                d.setdefault(int(di), []).append(uid)
        self.doc_uids = d

    def docs_of_uids(self, uids) -> set[int]:
        """Distinct documents that contain any of these unique sentences."""
        out: set[int] = set()
        for u in uids:
            out.update(self.uid_docs[u])
        return out

    # ---- builders -------------------------------------------------------- #
    @classmethod
    def from_clustering(cls, unique_index, embeddings, doc_texts: dict[int, str],
                        n_docs: int | None = None) -> "RunContext":
        """Build in-memory from a UniqueIndex + embeddings + per-document text."""
        if n_docs is None:
            n_docs = len(doc_texts) if doc_texts else (
                max((d for docs in unique_index.doc_idxs for d in docs), default=-1) + 1
            )
        return cls(
            E=np.asarray(embeddings, dtype=np.float32),
            texts=list(unique_index.texts),
            uid_docs=[sorted(set(d)) for d in unique_index.doc_idxs],
            n_docs=int(n_docs),
            doc_texts=doc_texts,
        )

    @classmethod
    def from_result(cls, result, doc_texts: dict[int, str] | None = None) -> "RunContext":
        """Build straight from a storeless :class:`~flashtopic.pipeline.ClusterResult`.

        Document texts are reconstructed from the result's sentence records unless given.
        Fully in-memory — no disk, no store.
        """
        dt = doc_texts if doc_texts is not None else _doc_texts_from_records(result.records)
        return cls.from_clustering(result.unique_index, result.embeddings, dt, n_docs=result.n_docs)

    @classmethod
    def from_store(cls, store, set_id: str, *, where=None) -> "RunContext":
        """Build from a :class:`~flashtopic.store.VectorStore` set (e.g. ``DuckDBStore``).

        ``where`` is a backend-specific filter passed through to ``get_unique_index`` (and
        the document-text reconstruction), so you can scope a run to a subset of atoms.
        """
        ui, E = store.get_unique_index(set_id, where=where)
        get_doc_texts = getattr(store, "doc_texts", None)
        doc_texts = get_doc_texts(set_id) if get_doc_texts is not None else {}
        return cls.from_clustering(ui, E, doc_texts)

    @classmethod
    def from_workspace(cls, ws, run_id: str) -> "RunContext":
        """Load a run's on-disk artifacts (embeddings + membership + document texts)."""
        E = np.load(ws.embeddings_path(run_id))
        texts, uid_docs = [], []
        for line in ws.body_membership_path(run_id).read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            texts.append(r["text"])
            uid_docs.append(sorted(set(r["doc_idxs"])))
        doc_texts = _load_doc_texts(ws, run_id)
        n_docs = len(doc_texts) if doc_texts else (
            max((d for docs in uid_docs for d in docs), default=-1) + 1
        )
        return cls(E=E, texts=texts, uid_docs=uid_docs, n_docs=n_docs, doc_texts=doc_texts)


def _load_doc_texts(ws, run_id: str) -> dict[int, str]:
    """Reconstruct each document's text from the run's sentences.jsonl (sentence order)."""
    path = ws.sentences_path(run_id)
    if not path.exists():
        return {}
    by_doc: dict[int, list[tuple[int, str]]] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        by_doc.setdefault(r["doc_idx"], []).append((r["sentence_idx"], r["text"]))
    return {d: "\n".join(t for _, t in sorted(s)) for d, s in by_doc.items()}


def _doc_texts_from_records(records) -> dict[int, str]:
    """In-memory twin of :func:`_load_doc_texts`: rebuild doc texts from sentence records."""
    by_doc: dict[int, list[tuple[int, str]]] = {}
    for r in records:
        by_doc.setdefault(r.doc_idx, []).append((r.sentence_idx, r.text))
    return {d: "\n".join(t for _, t in sorted(s)) for d, s in by_doc.items()}
