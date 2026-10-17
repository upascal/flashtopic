"""Entity alias clustering: merge near-duplicate entity norms into canonical groups.

NER produces one key per distinct string, so "US" / "U.S." / "United States" and
"Curie" / "Marie Curie" count as unrelated entities. This module folds a corpus's
per-text entity lists (``[{surface, label, norm}, ...]`` as produced by
:mod:`flashtopic.ner`) into an alias map ``{norm: canonical_norm}`` using three
deterministic, rule-based passes over a union-find:

1. acronym ↔ expansion (ORG/EVENT/GPE-LOC): "UN" → "united nations", but only when
   exactly one expansion in the label block matches the initials;
2. PERSON contiguous-suffix containment: "curie" → "marie curie", but only when the
   suffix matches exactly one longer-name cluster (an ambiguous "curie" that could be
   Marie or Pierre stays unmerged);
3. fuzzy spelling variants via rapidfuzz — Indel ratio ≥ 92 AND edit distance ≤ 2 AND
   min length ≥ 5, blocked by label. Jaro-Winkler is deliberately avoided (its prefix
   bonus merges "austria"/"australia"-class pairs).

DATE norms and anything containing a digit are excluded from every rule ("1942" vs
"1943" is edit distance 1), and pairs differing only in a trailing roman numeral
("world war i" vs "world war ii", "elizabeth i" vs "elizabeth ii") never fuzzy-merge.

Pure functions, no I/O: callers persist the map and apply it at aggregation time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Higher priority = wins label ties for a norm seen under several labels.
_LABEL_PRIORITY = ("PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "DATE")
_LABEL_RANK = {lab: i for i, lab in enumerate(_LABEL_PRIORITY)}

# GPE and LOC share a fuzzy/acronym block (spaCy flips between them constantly).
_BLOCK_OF = {"GPE": "GPE/LOC", "LOC": "GPE/LOC"}
_ACRONYM_BLOCKS = {"ORG", "EVENT", "GPE/LOC"}
_ACRO_STOPWORDS = frozenset({"of", "the", "and", "for", "in"})

_FUZZY_RATIO = 92.0
_FUZZY_MAX_EDITS = 2
_FUZZY_MIN_LEN = 5
_TOKEN_SORT_RATIO = 95.0


@dataclass
class EntityStats:
    """Corpus-wide tallies for one norm: texts containing it, surface + label votes."""

    df: int = 0
    surfaces: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)


def aggregate_entities(entity_lists: list[list[dict] | None]) -> dict[str, EntityStats]:
    """Fold per-text entity lists into ``{norm: EntityStats}``.

    Each text's list is already deduped by norm (the :func:`flashtopic.ner.extract_entities`
    contract), so incrementing per entry gives df = number of texts, matching the
    downstream aggregation semantics.
    """
    stats: dict[str, EntityStats] = {}
    for ents in entity_lists or []:
        for e in ents or []:
            norm = e.get("norm")
            if not norm:
                continue
            st = stats.setdefault(norm, EntityStats())
            st.df += 1
            st.surfaces[e.get("surface") or norm] += 1
            st.labels[e.get("label") or ""] += 1
    return stats


def _resolve_label(labels: Counter) -> str:
    """Majority label; ties broken by the fixed priority order (deterministic)."""
    if not labels:
        return ""
    return min(labels, key=lambda lab: (-labels[lab], _LABEL_RANK.get(lab, len(_LABEL_RANK)), lab))


def _has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


_ROMAN = frozenset(
    "i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx".split()
)


def _numbered_variants(a: str, b: str) -> bool:
    """True when the two norms differ only in a trailing roman numeral —
    "world war i"/"world war ii", "elizabeth i"/"elizabeth ii": distinct entities
    that sit at edit distance 1 and must never fuzzy-merge."""
    ta, tb = a.split(), b.split()
    return (
        ta[:-1] == tb[:-1]
        and ta[-1] != tb[-1]
        and (ta[-1] in _ROMAN or tb[-1] in _ROMAN)
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic root regardless of union order.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _initials(norm: str) -> str:
    return "".join(t[0] for t in norm.split() if t and t not in _ACRO_STOPWORDS)


def _acronym_edges(block: list[str], stats: dict[str, EntityStats], uf: _UnionFind) -> None:
    """Merge single-token uppercase acronyms into their unique multi-token expansion."""
    expansions_by_initials: dict[str, list[str]] = {}
    for nm in block:
        if len(nm.split()) >= 2 and not _has_digit(nm):
            expansions_by_initials.setdefault(_initials(nm), []).append(nm)
    for nm in block:
        if " " in nm or not (2 <= len(nm) <= 6) or _has_digit(nm):
            continue
        surface = stats[nm].surfaces.most_common(1)[0][0]
        if not (surface.isalpha() and surface.isupper()):
            continue
        matches = expansions_by_initials.get(nm, [])
        roots = {uf.find(m) for m in matches}
        if len(roots) == 1:
            uf.union(nm, matches[0])


def _person_suffix_edges(block: list[str], uf: _UnionFind) -> None:
    """Merge a PERSON name that is a contiguous token-suffix of exactly one longer name."""
    toks_of = {nm: tuple(nm.split()) for nm in block}
    # Shortest names first so "marie frank" can attach to "annelies marie frank"
    # before "frank" is considered against the merged cluster.
    for nm in sorted(block, key=lambda n: (len(toks_of[n]), n)):
        toks = toks_of[nm]
        matches = [
            other
            for other in block
            if len(toks_of[other]) > len(toks) and toks_of[other][-len(toks):] == toks
        ]
        roots = {uf.find(m) for m in matches}
        if len(roots) == 1:  # ambiguous suffixes ("curie" ⊂ marie+pierre) stay unmerged
            uf.union(nm, matches[0])


def _fuzzy_edges(block: list[str], label: str, uf: _UnionFind) -> None:
    """Spelling-variant merges: Indel ratio + edit-distance + length guards.

    The all-pairs ratio matrix runs through rapidfuzz's C ``cdist`` (multi-threaded,
    uint8 output) so a few-thousand-norm block stays sub-second; only the sparse
    above-cutoff pairs reach Python for the edit-distance confirmation.
    """
    import numpy as np
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import Levenshtein

    cands = [nm for nm in block if len(nm) >= _FUZZY_MIN_LEN and not _has_digit(nm)]
    if len(cands) >= 2:
        scores = process.cdist(
            cands, cands, scorer=fuzz.ratio, score_cutoff=_FUZZY_RATIO,
            dtype=np.uint8, workers=-1,
        )
        for i, j in zip(*np.nonzero(np.triu(scores, k=1))):
            a, b = cands[int(i)], cands[int(j)]
            if _numbered_variants(a, b):
                continue
            # Spelling variants share their first characters; contrast pairs that
            # clear the ratio gate differ at the START ("western europe" vs
            # "eastern europe", "northern"/"southern") — reject those.
            if a[:2] != b[:2]:
                continue
            if Levenshtein.distance(a, b, score_cutoff=_FUZZY_MAX_EDITS) <= _FUZZY_MAX_EDITS:
                uf.union(a, b)
    if label == "PERSON":
        multi = [nm for nm in cands if " " in nm]
        if len(multi) >= 2:
            scores = process.cdist(
                multi, multi, scorer=fuzz.token_sort_ratio,
                score_cutoff=_TOKEN_SORT_RATIO, dtype=np.uint8, workers=-1,
            )
            for i, j in zip(*np.nonzero(np.triu(scores, k=1))):
                a, b = multi[int(i)], multi[int(j)]
                if not _numbered_variants(a, b):
                    uf.union(a, b)


def build_alias_map(stats: dict[str, EntityStats]) -> dict[str, str]:
    """``{norm: canonical_norm}`` — non-identity entries only.

    Canonical per cluster: most tokens → highest df → shortest string →
    lexicographically smallest ("marie curie" beats "curie"; the correctly spelled
    variant of a typo pair usually wins on df).
    """
    label_of = {nm: _resolve_label(st.labels) for nm, st in stats.items()}
    blocks: dict[str, list[str]] = {}
    for nm, lab in label_of.items():
        if lab == "DATE":
            continue  # dates never merge — "1942"/"1943" are edit distance 1
        blocks.setdefault(_BLOCK_OF.get(lab, lab), []).append(nm)

    uf = _UnionFind()
    for key in sorted(blocks):
        block = sorted(blocks[key])
        if key in _ACRONYM_BLOCKS:
            _acronym_edges(block, stats, uf)
        if key == "PERSON":
            _person_suffix_edges(block, uf)
        _fuzzy_edges(block, key, uf)

    clusters: dict[str, list[str]] = {}
    for nm in (nm for block in blocks.values() for nm in block):
        clusters.setdefault(uf.find(nm), []).append(nm)

    out: dict[str, str] = {}
    for members in clusters.values():
        if len(members) < 2:
            continue
        canon = min(
            members, key=lambda nm: (-len(nm.split()), -stats[nm].df, len(nm), nm)
        )
        for nm in members:
            if nm != canon:
                out[nm] = canon
    return out
