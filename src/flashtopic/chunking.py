"""Markdown-aware document chunking primitives.

`parse_sections` + `split_list_units` + `_split_sentences_spacy` are the building
blocks the sentence corpus is made of; `chunk_{paragraph,sentence,semantic}` are
higher-level chunkers callers can use directly on any document. All variants strip
markdown and propagate the section path. Topic-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# spaCy is loaded lazily so importing this module is cheap. Sentence segmentation only
# needs the parser, so the small English pipeline is the default (the transformer model
# buys no accuracy here) — override it via `set_spacy_model` if you really need to.
_NLP = None
_SPACY_MODEL = "en_core_web_sm"


def set_spacy_model(name: str) -> None:
    """Override the spaCy model used for sentence segmentation (resets the cache)."""
    global _SPACY_MODEL, _NLP
    _SPACY_MODEL, _NLP = name, None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load(
            _SPACY_MODEL, disable=["ner", "tagger", "lemmatizer", "attribute_ruler"]
        )
    return _NLP


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)

MIN_CHUNK_WORDS = 25
MIN_SENTENCE_WORDS = 5
SENTENCES_PER_FALLBACK_CHUNK = 3


@dataclass
class Section:
    section_path: tuple[str, ...]
    body: str


@dataclass
class Chunk:
    text: str
    section_path: tuple[str, ...] = field(default_factory=tuple)
    doc_id: int = -1


def parse_sections(text: str) -> list[Section]:
    """Split markdown text into sections keyed by their heading path."""
    sections: list[Section] = []
    path_stack: list[tuple[int, str]] = []
    cursor = 0
    body_buf: list[str] = []

    def flush():
        body = "\n".join(body_buf).strip()
        if body:
            sections.append(
                Section(section_path=tuple(t for _, t in path_stack), body=body)
            )
        body_buf.clear()

    for m in HEADING_RE.finditer(text):
        pre = text[cursor : m.start()]
        if pre.strip():
            body_buf.append(pre)
        flush()
        level = len(m.group(1))
        title = m.group(2).strip()
        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, title))
        cursor = m.end()

    tail = text[cursor:]
    if tail.strip():
        body_buf.append(tail)
    flush()
    return sections


def _split_paragraphs(body: str) -> list[str]:
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in body.split("\n") if p.strip()]
    return paras


def _backward_merge(parts: list[str], min_words: int) -> list[str]:
    """Greedy: accumulate parts until min_words reached; tail merges into last."""
    merged: list[str] = []
    buf = ""
    for p in parts:
        buf = f"{buf} {p}".strip() if buf else p
        if len(buf.split()) >= min_words:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = f"{merged[-1]} {buf}"
        else:
            merged.append(buf)
    return merged


def _split_sentences_spacy(text: str) -> list[str]:
    nlp = _get_nlp()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


# List-aware pre-splitting: markdown bullets / inline enumerators pack several
# claims into one spaCy sentence; split at those boundaries before sentence
# splitting. Em-dash is excluded from the bullet class (it's prose punctuation).
_LINE_BULLET = re.compile(r"(?m)(?:^|\n)\s*[\-\*•]\s+")
_INLINE_BULLET = re.compile(r"\s+[\-\*•]\s+(?=[\*_\"']*[A-Z0-9\"'])")
_INLINE_ENUM = re.compile(r"\s+(?=\d{1,2}[\.\)]\s+[\*_\"']*[A-Z])")
# Lettered enumerators ("A. First point. B. Second point.") — anchored to a
# preceding sentence-final punctuation mark so a mid-name initial ("told J. Robert
# Oppenheimer") never splits. Without this, spaCy keeps "… World War II. B." as one
# sentence and NER spans leak across the item boundary.
_INLINE_ENUM_ALPHA = re.compile(r"(?<=[.!?:;])\s+(?=[A-Za-z][\.\)]\s+[\*_\"']*[A-Z])")
_LIST_BREAK = "\n<<<LIST_BREAK>>>\n"


def split_list_units(body_text: str) -> list[str]:
    """Split text at markdown/list boundaries before sentence splitting."""
    text = body_text.replace("\r\n", "\n").replace("\r", "\n")
    text = _LINE_BULLET.sub(_LIST_BREAK, text)
    text = _INLINE_BULLET.sub(_LIST_BREAK, text)
    text = _INLINE_ENUM.sub(_LIST_BREAK, text)
    text = _INLINE_ENUM_ALPHA.sub(_LIST_BREAK, text)
    return [p.strip() for p in text.split(_LIST_BREAK) if p.strip()]


def chunk_paragraph(
    text: str, doc_id: int = -1, min_words: int = MIN_CHUNK_WORDS
) -> list[Chunk]:
    """V0 — paragraph chunker (heading-aware)."""
    out: list[Chunk] = []
    for sec in parse_sections(text):
        paras = _split_paragraphs(sec.body)
        if len(paras) <= 1:
            sents = (
                _split_sentences_spacy(sec.body)
                if _spacy_available()
                else _regex_sentences(sec.body)
            )
            paras = [
                " ".join(sents[i : i + SENTENCES_PER_FALLBACK_CHUNK])
                for i in range(0, len(sents), SENTENCES_PER_FALLBACK_CHUNK)
            ]
        for chunk_text in _backward_merge(paras, min_words):
            out.append(Chunk(text=chunk_text, section_path=sec.section_path, doc_id=doc_id))
    return out


def chunk_sentence(
    text: str, doc_id: int = -1, min_words: int = MIN_SENTENCE_WORDS
) -> list[Chunk]:
    """V1 — one chunk per sentence; short tails merge backward (list-aware)."""
    out: list[Chunk] = []
    for sec in parse_sections(text):
        sents: list[str] = []
        for unit in split_list_units(sec.body):
            sents.extend(_split_sentences_spacy(unit))
        if not sents:
            continue
        for chunk_text in _backward_merge(sents, min_words):
            out.append(Chunk(text=chunk_text, section_path=sec.section_path, doc_id=doc_id))
    return out


def chunk_semantic(
    text: str,
    sentence_encoder: Callable[[Sequence[str]], np.ndarray],
    doc_id: int = -1,
    tau: float = 0.6,
    min_words: int = MIN_SENTENCE_WORDS,
) -> list[Chunk]:
    """V2 — sentence chunks merged when consecutive embeddings cosine > tau."""
    out: list[Chunk] = []
    for sec in parse_sections(text):
        sents = _split_sentences_spacy(sec.body)
        if not sents:
            continue
        if len(sents) == 1:
            out.append(Chunk(text=sents[0], section_path=sec.section_path, doc_id=doc_id))
            continue
        embs = sentence_encoder(sents)
        runs: list[list[str]] = [[sents[0]]]
        for i in range(1, len(sents)):
            if float(np.dot(embs[i - 1], embs[i])) >= tau:
                runs[-1].append(sents[i])
            else:
                runs.append([sents[i]])
        for chunk_text in _backward_merge([" ".join(r) for r in runs], min_words):
            out.append(Chunk(text=chunk_text, section_path=sec.section_path, doc_id=doc_id))
    return out


def _spacy_available() -> bool:
    try:
        _get_nlp()
        return True
    except Exception:
        return False


def _regex_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def section_label(path: tuple[str, ...] | Sequence[str]) -> str:
    return " > ".join(path)
