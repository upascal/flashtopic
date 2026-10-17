"""Load documents from JSONL / JSONL.gz into ``Doc`` records.

Deliberately light and store-free: the rest of the pipeline works on ``Doc`` lists
and on-disk run artifacts, so ingestion just maps records → ``Doc(doc_id, text)``.
Field names are configurable with dotted paths (e.g. ``body.text`` or
``meta.completion_id``) so the same loader handles arbitrary corpora, not just one schema.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable

from flashtopic.schema import Doc


def stream_jsonl(path: str | Path) -> Iterable[dict]:
    """Yield one parsed dict per line from a .jsonl or .jsonl.gz file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _dig(rec: dict, dotted: str):
    """Fetch a (possibly nested) field by dotted path; None if any hop is missing."""
    cur = rec
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def to_docs(
    records: Iterable[dict],
    *,
    text_field: str = "text",
    id_field: str | None = None,
    exclude: tuple[str, str] | None = None,
    dedup: bool = True,
) -> list[Doc]:
    """Map raw records → ``Doc`` list.

    ``text_field`` / ``id_field`` are dotted paths. The default ``text_field="text"`` suits
    a generic corpus; if that field is missing it falls back to a top-level ``completion``
    (so LLM-completion records load without extra flags). ``id_field`` defaults to ``None``
    → a content-hash id; pass e.g. ``"meta.completion_id"`` for a stable external id.
    ``exclude=(dotted_field, value)`` drops records whose field equals ``value`` (e.g.
    ``("meta.run_id", "0")`` to skip a pilot run).
    """
    docs: list[Doc] = []
    seen: set[str] = set()
    for rec in records:
        if exclude is not None and str(_dig(rec, exclude[0])) == str(exclude[1]):
            continue
        text = _dig(rec, text_field)
        if text is None:
            text = rec.get("completion")
        if not text:
            continue
        doc_id = _dig(rec, id_field) if id_field else None
        if doc_id is None:
            doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        doc_id = str(doc_id)
        if dedup and doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(Doc(doc_id=doc_id, text=text))
    return docs


def load_docs(path: str | Path, **kwargs) -> list[Doc]:
    """Stream a JSONL(.gz) file and map it to ``Doc`` records (see ``to_docs``)."""
    docs = to_docs(stream_jsonl(path), **kwargs)
    if not docs:
        raise ValueError(f"no documents loaded from {path}")
    return docs
