"""Sentence-level corpus construction.

Sentence-first but list-aware: markdown bullets and inline enumerators often pack
several claims into one spaCy sentence, so we split at those list boundaries before
sentence splitting, then merge short label stubs forward (an item like
"Restrictions:" stays attached to its claim).

Fragments — table header/separator rows, and short pieces that can't merge into a
sentence of their own list item — are kept as ``kind="fragment"`` records rather than dropped,
so nothing is lost, but they never enter the ``body`` stream that gets clustered.

Generalized from the ancestor repo: documents (not "essays") are the unit, so the
per-sentence document ordinal is ``doc_idx`` and the unique-index membership maps to
``doc_idxs``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from flashtopic.chunking import (
    _split_sentences_spacy,
    parse_sections,
    split_list_units,
)
from flashtopic.schema import Doc

MIN_WORDS = 4

_BOLD = re.compile(r"\*\*.*?\*\*|__.*?__")
_ATX = re.compile(r"^\s*#{1,6}\s+")
_LEAD = re.compile(r"^\s*[>#\-\*•–—]+\s*")
_LEAD_NUM = re.compile(r"^\s*\d+[\.\)]\s*")


@dataclass
class SentenceRecord:
    sentence_idx: int  # global index across the whole corpus
    doc_idx: int  # which document (0-based ordinal)
    doc_id: str  # Doc.doc_id
    section_path: tuple[str, ...]
    kind: str  # "body" | "heading" | "fragment"
    text: str
    n_words: int


def strip_markup(raw: str) -> str:
    s = _LEAD.sub("", raw)
    s = _LEAD_NUM.sub("", s)
    s = s.replace("**", "").replace("__", "")
    return " ".join(s.split())


def classify_sentence(raw: str) -> tuple[str, str]:
    """Return (kind, clean_text). Heading iff pure markdown markup."""
    raw = raw.strip()
    if _ATX.match(raw):
        return "heading", strip_markup(raw)
    residue = _BOLD.sub("", raw)
    residue = re.sub(r"^\s*[>#\-\*•\d\.\)\s]+", "", residue)
    residue = re.sub(r"[\-\s:]+", " ", residue).strip()
    if residue == "":
        return "heading", strip_markup(raw)
    return "body", strip_markup(raw)


_TABLE_ROW = re.compile(r"^\s*\|.*\|?\s*$|^\s*\S.*\|.*\|")


def is_table_row(text: str) -> bool:
    """Markdown table row or separator (``| a | b |``, ``|---|---|``, ``a | b | c``)."""
    return bool(_TABLE_ROW.match(text.strip()))


_MIN_TABLE_ROW_WORDS = 8  # below this a row is a header/label, not a claim


def table_row_text(row: str) -> str:
    """``| American democracy | Explain natural rights |`` → ``American democracy: Explain
    natural rights``. Separator rows (``|---|:--:|``) come back empty."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    cells = [c for c in cells if c and not re.fullmatch(r":?-{2,}:?", c)]
    if not cells:
        return ""
    return strip_markup(cells[0] + (": " + " ".join(cells[1:]) if len(cells) > 1 else ""))


def _split_table_lines(unit: str) -> list[tuple[bool, str]]:
    """Separate table lines from prose *before* spaCy, which would glue a table and the
    text after it into one "sentence". Returns ``(is_table, text)`` runs in order."""
    runs: list[tuple[bool, list[str]]] = []
    for line in unit.split("\n"):
        t = is_table_row(line) if line.strip() else None
        if t is None:
            if runs:
                runs[-1][1].append(line)
            continue
        if runs and runs[-1][0] == t:
            runs[-1][1].append(line)
        else:
            runs.append((t, [line]))
    out = []
    for is_tab, lines in runs:
        if is_tab:
            out += [(True, ln.strip()) for ln in lines if ln.strip()]
        elif "\n".join(lines).strip():
            out.append((False, "\n".join(lines).strip()))
    return out


_LEAD_BULLETNUM = re.compile(r"^\s*(?:[-*•]\s+|\d+[.\)]\s+)?")
_BOLD_AT_START = re.compile(r"^(\*\*.+?\*\*|__.+?__)", re.DOTALL)
_HEADER_SEP = re.compile(r"\s*[:—–]\s*|\s+-\s+|\s*\n")


def _split_leading_header(rs: str) -> tuple[str | None, str]:
    """Peel a leading bold heading off `rs` (a label, not in-text emphasis)."""
    after = rs[_LEAD_BULLETNUM.match(rs).end() :]
    m = _BOLD_AT_START.match(after)
    if not m:
        return None, rs
    tail = after[m.end() :]
    if tail.strip() == "":
        return strip_markup(m.group(1)), ""
    sep = _HEADER_SEP.match(tail)
    if not sep:
        return None, rs
    return strip_markup(m.group(1)), tail[sep.end() :].strip()


def _is_label_stub(text: str) -> bool:
    words = text.split()
    return (
        text.endswith(":")
        and 1 <= len(words) <= 8
        and not any(ch in text[:-1] for ch in ".!?")
    )


def _process_section(body_text: str, min_words: int) -> list[tuple[str, str]]:
    """Split one section into (kind, clean_text).

    Short pieces merge forward into the next sentence of the *same* list unit. A label
    stub ("Restrictions:") still carries across units to attach to its claim, but any
    other short piece left at the end of a unit merges back into that unit's previous
    sentence or, failing that, becomes a ``fragment`` — it never glues onto an
    unrelated list item, and never surfaces as a ``body`` sentence of its own.
    """
    out: list[tuple[str, str]] = []
    carry = ""
    carry_is_stub = False

    def flush_carry():
        nonlocal carry, carry_is_stub
        if carry:
            out.append(("fragment", carry))
        carry, carry_is_stub = "", False

    for unit in split_list_units(body_text):
        unit_start = len(out)
        pieces: list[str] = []
        for is_tab, text in _split_table_lines(unit):
            if is_tab:
                pieces.append(text)  # a table row is its own piece; flagged below
            else:
                pieces += _split_sentences_spacy(text)
        for rs in pieces:
            header, rs = _split_leading_header(rs)
            while header is not None:
                flush_carry()
                out.append(("heading", header))
                header, rs = _split_leading_header(rs)
            if not rs.strip():
                continue
            if is_table_row(rs):
                # A content row reads as a sentence ("American democracy: Explain …");
                # header and separator rows ("| Area | Needed treatment |") are fragments.
                flush_carry()
                row = table_row_text(rs)
                if len(row.split()) >= _MIN_TABLE_ROW_WORDS:
                    out.append(("body", row))
                elif row:
                    out.append(("fragment", row))
                continue
            kind, text = classify_sentence(rs)
            if not text:
                continue
            if kind == "heading":
                flush_carry()
                out.append(("heading", text))
                continue
            if _is_label_stub(text):
                carry = f"{carry} {text}".strip() if carry else text
                carry_is_stub = True
                continue
            text = f"{carry} {text}".strip() if carry else text
            carry, carry_is_stub = "", False
            if len(text.split()) < min_words:
                carry = text
                continue
            out.append(("body", text))
        if carry and not carry_is_stub:
            if len(out) > unit_start and out[-1][0] == "body":
                out[-1] = ("body", f"{out[-1][1]} {carry}")
                carry = ""
            else:
                flush_carry()
    if carry:
        if carry_is_stub and out and out[-1][0] == "body":
            out[-1] = ("body", f"{out[-1][1]} {carry}")
            carry = ""
        flush_carry()
    return out


def build_sentence_corpus(
    docs: list[Doc], min_words: int = MIN_WORDS
) -> list[SentenceRecord]:
    """Split every Doc into tagged, markup-stripped sentence records."""
    records: list[SentenceRecord] = []
    for doc_idx, doc in enumerate(docs):
        for sec in parse_sections(doc.text):
            for kind, text in _process_section(sec.body, min_words):
                records.append(
                    SentenceRecord(
                        sentence_idx=len(records),
                        doc_idx=doc_idx,
                        doc_id=doc.doc_id,
                        section_path=sec.section_path,
                        kind=kind,
                        text=text,
                        n_words=len(text.split()),
                    )
                )
    return records


@dataclass
class UniqueIndex:
    texts: list[str]
    sentence_idxs: list[list[int]]
    doc_idxs: list[list[int]]

    @property
    def counts(self) -> list[int]:
        return [len(s) for s in self.sentence_idxs]


def build_unique_index(records: list[SentenceRecord], kind: str) -> UniqueIndex:
    """Deduplicate one stream's texts → unique list + membership map."""
    order: dict[str, int] = {}
    texts: list[str] = []
    sids: list[list[int]] = []
    dids: list[list[int]] = []
    for r in records:
        if r.kind != kind:
            continue
        u = order.get(r.text)
        if u is None:
            u = len(texts)
            order[r.text] = u
            texts.append(r.text)
            sids.append([])
            dids.append([])
        sids[u].append(r.sentence_idx)
        dids[u].append(r.doc_idx)
    return UniqueIndex(texts=texts, sentence_idxs=sids, doc_idxs=dids)
