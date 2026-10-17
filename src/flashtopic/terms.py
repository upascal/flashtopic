"""Distinctive-term scoring — cheap, model-free cluster labels.

For each cluster we surface the uni/bigrams that most distinguish it from the rest
of the corpus (the sibling pool), one-vs-rest on per-sentence presence counts. Three
pluggable scorers behind a common ``TermScorer`` signature:

* ``log_odds`` (default) — weighted **log-odds-ratio with an informative Dirichlet
  prior** (Monroe et al. "Fightin' Words"). Variance-normalized, so rare flukes don't
  spike and sibling-shared words are discounted. The most robust at *distinguishing*
  near-duplicate clusters.
* ``ctfidf`` — BERTopic-style class-based TF-IDF (one-vs-rest). Frequency-driven;
  surfaces what's *common* in the cluster. Provided for A/B comparison.
* ``tfidf`` — legacy TF × weak-sibling-IDF; most *representative* of the cluster bulk,
  weakest at distinguishing.

Domain stopwords are passed in (from ``Workspace.domain_stopwords``) — no corpus-specific
defaults are baked in here.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

import numpy as np

# Structural English noise only. Domain words come from the caller via
# `extra_stopwords`, not from this module.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "by", "at", "from", "that", "this", "these", "those", "it", "its",
    "was", "were", "is", "are", "be", "been", "being", "had", "has", "have",
    "their", "they", "them", "his", "her", "he", "she", "which", "who", "whom",
    "also", "would", "could", "should", "while", "during", "such", "into",
    "not", "no", "than", "then", "there", "here", "more", "most", "some", "other",
}
_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")

# A scorer maps (member presence-counts, sibling presence-counts, totals) → ranked terms.
TermScorer = Callable[..., list[tuple[float, str]]]


def _make_tokenizer(extra_stopwords: Iterable[str] | None):
    stop = set(_STOPWORDS)
    if extra_stopwords:
        stop |= {w.lower() for w in extra_stopwords}

    def tokenize(text: str) -> list[str]:
        toks = [t for t in _TOKEN_RE.findall(text.lower()) if t not in stop]
        return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]

    return tokenize


def _presence_counts(
    sentences: list[str], tokenize, weights: list[float] | None = None
) -> dict[str, float]:
    """Per-sentence presence counts, optionally weighted.

    ``weights[i]`` scales sentence ``i``'s contribution — pass the number of *distinct
    documents* a (deduplicated) sentence appears in so a phrase echoed across the corpus
    outweighs a single document's one-off vocabulary. Default weight 1.0 reproduces the
    old unweighted behavior.
    """
    counts: dict[str, float] = {}
    for i, s in enumerate(sentences):
        w = 1.0 if weights is None else float(weights[i])
        for g in set(tokenize(s)):  # per-sentence presence, not raw frequency
            counts[g] = counts.get(g, 0.0) + w
    return counts


def _score_log_odds(member: dict[str, int], sibling: dict[str, int]) -> list[tuple[float, str]]:
    """Monroe weighted log-odds-ratio with an informative Dirichlet prior."""
    nA = max(1, sum(member.values()))
    nB = max(1, sum(sibling.values()))
    a0 = nA + nB
    out: list[tuple[float, str]] = []
    for g, yA in member.items():
        if yA < 2:  # ignore singletons — too noisy
            continue
        yB = sibling.get(g, 0)
        a_w = yA + yB  # informative prior = combined count
        num_a, den_a = yA + a_w, max(1e-9, nA + a0 - yA - a_w)
        num_b, den_b = yB + a_w, max(1e-9, nB + a0 - yB - a_w)
        delta = np.log(num_a / den_a) - np.log(num_b / den_b)
        sigma2 = 1.0 / (yA + a_w) + 1.0 / (yB + a_w)
        z = delta / np.sqrt(sigma2)
        if z > 0:
            out.append((float(z), g))
    return out


def _score_ctfidf(member: dict[str, int], sibling: dict[str, int]) -> list[tuple[float, str]]:
    """BERTopic-style class TF-IDF, one-vs-rest: tf · log(1 + A / f_t)."""
    nA, nB = max(1, sum(member.values())), max(1, sum(sibling.values()))
    avg_class = (nA + nB) / 2.0
    out: list[tuple[float, str]] = []
    for g, yA in member.items():
        if yA < 2:
            continue
        f_t = yA + sibling.get(g, 0)
        out.append((yA * float(np.log(1.0 + avg_class / f_t)), g))
    return out


def _score_tfidf(member: dict[str, int], sibling: dict[str, int],
                 n_member: int, n_sibling: int) -> list[tuple[float, str]]:
    """Legacy TF × weak-sibling-IDF."""
    out: list[tuple[float, str]] = []
    for g, mc in member.items():
        if mc < 2:
            continue
        tf = mc / max(1, n_member)
        sib = sibling.get(g, 0) / max(1, n_sibling)
        out.append((tf * float(np.log(2.0 / (1.0 + sib))), g))
    return out


_METHODS = {"log_odds", "ctfidf", "tfidf"}


def distinctive_terms(
    member_sentences: list[str],
    sibling_sentences: list[str],
    *,
    n_terms: int = 12,
    extra_stopwords: Iterable[str] | None = None,
    method: str = "log_odds",
    member_weights: list[float] | None = None,
    sibling_weights: list[float] | None = None,
    member_doc_idxs: list[list[int]] | None = None,
    min_df: int = 2,
) -> list[str]:
    """Top distinctive uni/bigrams for this cluster vs the sibling pool (no model call).

    ``member_weights`` / ``sibling_weights`` weight each sentence's presence (see
    :func:`_presence_counts`). ``member_doc_idxs[i]`` is the list of document indices
    for member sentence ``i``; when given (and ``min_df > 1``), a term must appear in at
    least ``min_df`` *distinct documents* to be eligible — the document-level floor that
    stops a single verbose document from minting its own distinctive terms.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_METHODS)}")
    tokenize = _make_tokenizer(extra_stopwords)
    member = _presence_counts(member_sentences, tokenize, member_weights)
    sibling = _presence_counts(sibling_sentences, tokenize, sibling_weights)

    if member_doc_idxs is not None and min_df > 1:
        term_docs: dict[str, set[int]] = {}
        for sent, docs in zip(member_sentences, member_doc_idxs):
            for g in set(tokenize(sent)):
                term_docs.setdefault(g, set()).update(docs)
        member = {g: c for g, c in member.items() if len(term_docs.get(g, ())) >= min_df}

    if method == "log_odds":
        scored = _score_log_odds(member, sibling)
    elif method == "ctfidf":
        scored = _score_ctfidf(member, sibling)
    else:
        scored = _score_tfidf(member, sibling, len(member_sentences), len(sibling_sentences))

    scored.sort(reverse=True)
    return [g for _, g in scored[:n_terms]]
