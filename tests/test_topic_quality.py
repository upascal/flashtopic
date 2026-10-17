"""Topic quality inputs: fragment filtering, typical exemplars, cluster coherence."""

from __future__ import annotations

import numpy as np
import pytest

from flashtopic.prevalence.codebook import build_codebook
from flashtopic.prevalence.topics import Topic
from flashtopic.sampling import cluster_coherence, coverage_exemplars, typical_exemplars


def _has_spacy() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


needs_spacy = pytest.mark.skipif(not _has_spacy(), reason="spacy + en_core_web_sm required")


# --------------------------------------------------------------------------- #
# Fragments never reach the clustered body stream
# --------------------------------------------------------------------------- #
def _kinds(text):
    from flashtopic.corpus import _process_section

    return _process_section(text, 4)


def test_table_rows_are_fragments():
    from flashtopic.corpus import is_table_row

    assert is_table_row("| Area | Needed treatment |")
    assert is_table_row("|---|---|")
    assert is_table_row("Area | Needed treatment | Priority")
    assert not is_table_row("The museum should present evidence, not ideology.")


@needs_spacy
def test_table_and_short_pieces_become_fragments_not_body():
    out = _kinds(
        "The museum should apply one evidentiary standard to every exhibit.\n\n"
        "| Area | Needed treatment |\n|---|---|\n"
        "Spec"
    )
    body = [t for k, t in out if k == "body"]
    frags = [t for k, t in out if k == "fragment"]
    assert body == ["The museum should apply one evidentiary standard to every exhibit."]
    assert any("Area" in f for f in frags) and "Spec" in frags


@needs_spacy
def test_content_table_rows_read_as_sentences():
    out = _kinds("| Gallery | Change |\n|---|---|\n"
                 "| American democracy | Explain natural rights, federalism, and constitutional design. |\n")
    assert ("body", "American democracy: Explain natural rights, federalism, and constitutional design.") in out
    assert ("fragment", "Gallery: Change") in out


@needs_spacy
def test_short_list_item_does_not_glue_onto_the_next_item():
    out = _kinds("- Collective-guilt language\n- Euphemisms that conceal injustice in exhibits\n")
    body = [t for k, t in out if k == "body"]
    assert "Collective-guilt language Euphemisms that conceal injustice in exhibits" not in body
    assert "Euphemisms that conceal injustice in exhibits" in body
    assert ("fragment", "Collective-guilt language") in out


@needs_spacy
def test_label_stub_still_attaches_to_its_claim():
    out = _kinds("- Restrictions:\n- Exhibits must cite primary sources for every claim.\n")
    assert ("body", "Restrictions: Exhibits must cite primary sources for every claim.") in out


# --------------------------------------------------------------------------- #
# Typical vs coverage exemplars, coherence
# --------------------------------------------------------------------------- #
def _cluster(rng, n_core=20, n_out=3):
    core = rng.normal([1, 0, 0], 0.05, size=(n_core, 3))
    outl = rng.normal([0, 1, 0], 0.05, size=(n_out, 3))
    X = np.vstack([core, outl])
    return X / np.linalg.norm(X, axis=1, keepdims=True), n_core


def test_typical_exemplars_come_from_the_core_coverage_reaches_the_tail():
    X, n_core = _cluster(np.random.default_rng(0))
    idx = list(range(100, 100 + len(X)))
    typ = typical_exemplars(X, idx, 5, dedup_tau=0.9999)
    assert all(i - 100 < n_core for i in typ)
    cov = coverage_exemplars(X, idx, 5)
    assert any(i - 100 >= n_core for i in cov)  # the property that made them wrong for codebooks


def test_typical_exemplars_skip_near_duplicates():
    X = np.array([[1, 0, 0], [1, 1e-4, 0], [0.7, 0.7, 0]], dtype=float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert len(typical_exemplars(X, [0, 1, 2], 3, dedup_tau=0.99)) == 2


def test_coherence_tail_exposes_a_second_idea():
    rng = np.random.default_rng(1)
    tight, _ = _cluster(rng, n_core=30, n_out=0)
    mixed, _ = _cluster(rng, n_core=20, n_out=10)
    a, b = cluster_coherence(tight), cluster_coherence(mixed)
    assert a["median"] > 0.99 and b["p10"] < a["p10"] - 0.2


def test_codebook_prefers_typical_exemplars():
    t = Topic("T1", [0], label="evidentiary standards", exemplars=["| Area | Needed", "Spec"],
              typical_exemplars=["Apply the same evidentiary standard to every subject."])
    assert build_codebook(t).positive_examples == ["Apply the same evidentiary standard to every subject."]
    legacy = Topic("T2", [0], label="x", exemplars=["old coverage exemplar"])
    assert build_codebook(legacy).positive_examples == ["old coverage exemplar"]
