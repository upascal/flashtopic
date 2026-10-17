"""Tests for the portable (no-GPU, no-LLM) core: ingest, corpus, terms, sampling, NER."""

from __future__ import annotations

import numpy as np
import pytest

from flashtopic.corpus import build_sentence_corpus, build_unique_index
from flashtopic.ingest import to_docs
from flashtopic.sampling import coverage_exemplars
from flashtopic.schema import Doc
from flashtopic.terms import distinctive_terms


# ---- ingest ---------------------------------------------------------------- #
def test_to_docs_maps_dedups_and_excludes():
    # LLM-completion records: point text_field/id_field at the completion schema explicitly
    # (the defaults are source-neutral now).
    records = [
        {"meta": {"completion_id": "a", "run_id": "1"}, "completion": "first text"},
        {"meta": {"completion_id": "a", "run_id": "1"}, "completion": "dup id"},  # dedup
        {"meta": {"completion_id": "b", "run_id": "0"}, "completion": "pilot"},  # excluded
        {"meta": {"completion_id": "c", "run_id": "1"}, "completion": "third"},
    ]
    docs = to_docs(records, text_field="completion", id_field="meta.completion_id",
                   exclude=("meta.run_id", "0"))
    assert [d.doc_id for d in docs] == ["a", "c"]
    assert docs[0].text == "first text"


def test_to_docs_default_text_field_and_completion_fallback():
    # Default suits a generic corpus (top-level `text`); a bare completion record still
    # loads via the fallback, with a content-hash id.
    docs = to_docs([{"text": "generic doc"}, {"completion": "llm output"}])
    assert [d.text for d in docs] == ["generic doc", "llm output"]
    assert all(len(d.doc_id) == 16 for d in docs)


def test_to_docs_falls_back_to_content_hash():
    docs = to_docs([{"text": "no id here"}], id_field=None)
    assert len(docs) == 1 and len(docs[0].doc_id) == 16


# ---- corpus ---------------------------------------------------------------- #
def test_corpus_dedup_and_membership():
    docs = [
        Doc("d1", "# History\n\nThe shared sentence appears here. A unique tail follows it."),
        Doc("d2", "The shared sentence appears here. A different ending appears instead."),
    ]
    recs = build_sentence_corpus(docs)
    bodies = [r for r in recs if r.kind == "body"]
    # an ATX heading becomes a section_path, not a heading record
    assert any(r.section_path == ("History",) for r in bodies)
    ui = build_unique_index(recs, "body")
    # the shared sentence appears in both documents
    shared = [i for i, t in enumerate(ui.texts) if t.startswith("The shared sentence")]
    assert shared and sorted(ui.doc_idxs[shared[0]]) == [0, 1]
    assert len(bodies) == 4 and len(ui.texts) == 3


# ---- terms ----------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["log_odds", "ctfidf", "tfidf"])
def test_distinctive_terms_surface_the_distinguishing_word(method):
    members = ["the strike vote continued", "calls for a strike grew", "strike was debated"]
    siblings = ["the bus ran late", "a new train route opened", "the station was full"] * 3
    terms = distinctive_terms(members, siblings, n_terms=5, method=method)
    assert "strike" in terms


def test_extra_stopwords_suppress_domain_terms():
    members = ["transit strike debate", "transit strike calls", "transit strike vote"]
    siblings = ["transit fare change", "transit route opening"] * 3
    terms = distinctive_terms(members, siblings, n_terms=5, extra_stopwords={"transit"})
    assert "transit" not in " ".join(terms)
    assert "strike" in terms


def test_min_df_drops_single_response_vocab():
    # A rare term ("zorblax") appears in two member sentences but BOTH come from the same
    # response (doc 0); a genuinely shared term ("budget") spans three responses. With a
    # distinct-response floor, the one-off vocab must not surface while the shared one does.
    members = ["zorblax budget plan", "zorblax internal memo", "annual budget review", "budget shortfall"]
    member_doc_idxs = [[0], [0], [1], [2]]
    siblings = ["the weather was nice", "a quiet afternoon", "trains run on time"] * 3
    terms = distinctive_terms(
        members, siblings, n_terms=8, member_doc_idxs=member_doc_idxs, min_df=2
    )
    assert "budget" in terms
    assert "zorblax" not in terms
    # Without the floor (no doc idxs), the old sentence-level behavior still surfaces it.
    legacy = distinctive_terms(members, siblings, n_terms=8)
    assert "zorblax" in legacy


def test_response_weighting_boosts_widely_echoed_terms():
    # Two distinctive terms each clear the floor, but "echoed" appears across many more
    # responses than "rare"; response-weighting should rank it at least as high.
    members = ["echoed signal here", "echoed signal there", "echoed once more", "rare blip seen", "rare blip again"]
    member_doc_idxs = [[0], [1], [2], [3], [4]]
    weights = [float(len(set(d))) for d in member_doc_idxs]
    siblings = ["unrelated chatter", "off topic noise"] * 3
    terms = distinctive_terms(
        members, siblings, n_terms=4, member_weights=weights,
        member_doc_idxs=member_doc_idxs, min_df=2,
    )
    assert "echoed" in terms


# ---- sampling -------------------------------------------------------------- #
def test_coverage_exemplars_returns_n_distinct_spanning_indices():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    picks = coverage_exemplars(emb, list(range(1000, 1200)), n=20)
    assert len(picks) == 20
    assert len(set(picks)) == 20
    assert all(1000 <= p < 1200 for p in picks)


def test_coverage_exemplars_returns_all_when_small():
    emb = np.eye(5, dtype=np.float32)
    assert coverage_exemplars(emb, [0, 1, 2, 3, 4], n=10) == [0, 1, 2, 3, 4]


# ---- NER + distinctiveness (needs en_core_web_sm) -------------------------- #
def test_ner_extracts_person_and_place():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("en_core_web_sm not installed")
    from flashtopic.ner import extract_entities

    entity_lists, entity_df, *_ = extract_entities(
        ["Marie Curie won two Nobel prizes.", "Lyon hosted the conference in 1998."]
    )
    norms = {e["norm"] for ents in entity_lists for e in ents}
    assert "marie curie" in norms
    assert "lyon" in norms


def test_distinctiveness_lemmatizes_and_scores():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("en_core_web_sm not installed")
    from flashtopic.distinct import lemmatize_corpus, sentence_distinctiveness

    lemma_sets, surf, df = lemmatize_corpus(
        ["The films were watched widely.", "A rare hydra appeared once."]
    )
    rows = sentence_distinctiveness(lemma_sets, surf, df)
    # the corpus-unique word scores higher than the common one
    assert rows[1]["score"] >= rows[0]["score"]
