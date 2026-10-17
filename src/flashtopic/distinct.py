"""Per-sentence lexical distinctiveness (content-lemma rarity), computed once.

A sentence is "distinctive" when it contains a rare *word*. On a single-topic corpus,
inflected forms fragment counts ("watching"/"watched" each appear once → everything
looks rare), so we lemmatize with spaCy (collapsing to "watch") and keep only content
words (NOUN/PROPN/ADJ/VERB). An intrinsically unusual word survives at df=1 and pops;
mundane inflections do not. Query-agnostic — the core functions take a list of texts.
"""

from __future__ import annotations

import json
import math

from flashtopic.terms import _STOPWORDS

_CONTENT_POS = {"NOUN", "PROPN", "ADJ", "VERB"}
_DEFAULT_MODEL = "en_core_web_sm"
_NLP = None
_NLP_MODEL = None


def _nlp(model: str = _DEFAULT_MODEL):
    """Lazily load a spaCy pipeline with parser/NER disabled (need tagger + lemmatizer).

    Reloads if `model` changes (e.g. switching to `en_core_web_trf`).
    """
    global _NLP, _NLP_MODEL
    if _NLP is None or _NLP_MODEL != model:
        import spacy

        _NLP = spacy.load(model, disable=["parser", "ner"])
        _NLP_MODEL = model
    return _NLP


def lemmatize_corpus(
    texts: list[str],
    *,
    model: str = _DEFAULT_MODEL,
) -> tuple[list[list[str]], dict[str, str], dict[str, int]]:
    """Lemmatize each text → (lemma_sets, surface_by_lemma, lemma_df).

    Input is lowercased before tagging so Title-Case headings ("Films") are tagged as
    common nouns and lemmatized properly ("films"→"film") rather than kept as PROPN.
    """
    nlp = _nlp(model)
    lemma_sets: list[list[str]] = []
    surface_by_lemma: dict[str, str] = {}
    lemma_df: dict[str, int] = {}
    for doc in nlp.pipe((t.lower() for t in texts), batch_size=256):
        seen: dict[str, None] = {}
        for tok in doc:
            if not tok.is_alpha or tok.is_stop or tok.pos_ not in _CONTENT_POS:
                continue
            lemma = tok.lemma_.lower()
            if len(lemma) < 2 or lemma in _STOPWORDS:
                continue
            if lemma not in seen:
                seen[lemma] = None
                surface_by_lemma.setdefault(lemma, tok.text)
        lemmas = list(seen.keys())
        lemma_sets.append(lemmas)
        for lemma in lemmas:
            lemma_df[lemma] = lemma_df.get(lemma, 0) + 1
    return lemma_sets, surface_by_lemma, lemma_df


def sentence_distinctiveness(
    lemma_sets: list[list[str]],
    surface_by_lemma: dict[str, str],
    lemma_df: dict[str, int],
    *,
    n_terms: int = 8,
) -> list[dict]:
    """Per text: rarity of its rarest content lemma + top rare terms (score in [0,1])."""
    n = max(2, len(lemma_sets))
    log_n = math.log(n)
    out: list[dict] = []
    for lemmas in lemma_sets:
        ranked = sorted((lemma_df[lemma], lemma) for lemma in lemmas)  # ascending df
        if not ranked:
            out.append({"lemmas": [], "surfaces": [], "rarest_df": n, "score": 0.0, "terms": []})
            continue
        rarest_df = ranked[0][0]
        out.append({
            "lemmas": [lemma for _, lemma in ranked],
            "surfaces": [surface_by_lemma[lemma] for _, lemma in ranked],
            "rarest_df": rarest_df,
            "score": math.log(n / rarest_df) / log_n,
            "terms": [f"{surface_by_lemma[lemma]} ·{df}" for df, lemma in ranked[:n_terms]],
        })
    return out


def compute_distinctiveness(ws, run_id: str, *, model: str | None = None, n_terms: int = 8):
    """Compute per-unique distinctiveness for a run and write the artifacts.

    Reads unique texts from the run's body membership and writes
    ``sentence_distinctiveness.jsonl`` + ``lemma_df.json`` under the run's corpus dir.
    Uses ``ws.spacy_model`` unless ``model`` overrides it.
    """
    membership = [
        json.loads(line)
        for line in ws.body_membership_path(run_id).read_text().splitlines()
        if line
    ]
    texts = [m["text"] for m in membership]
    lemma_sets, surface_by_lemma, lemma_df = lemmatize_corpus(texts, model=model or ws.spacy_model)
    rows = sentence_distinctiveness(lemma_sets, surface_by_lemma, lemma_df, n_terms=n_terms)

    out_path = ws.distinctiveness_path(run_id)
    with out_path.open("w") as f:
        for uid, row in enumerate(rows):
            f.write(json.dumps({"unique_id": uid, **row}) + "\n")
    ws.lemma_df_path(run_id).write_text(json.dumps(lemma_df))
    return out_path
