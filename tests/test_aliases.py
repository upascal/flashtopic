"""Alias clustering: acronym/suffix/fuzzy merge rules, guards, and canonical picks."""

import pytest

pytest.importorskip("rapidfuzz")

from flashtopic.aliases import aggregate_entities, build_alias_map


def _ent(surface, label, norm=None):
    return {"surface": surface, "label": label, "norm": norm or surface.lower()}


def _stats(*texts):
    """Each arg is one text's entity list."""
    return aggregate_entities(list(texts))


def test_aggregate_counts_texts_not_mentions():
    stats = _stats(
        [_ent("Marie Curie", "PERSON")],
        [_ent("Marie Curie", "PERSON"), _ent("Lyon", "GPE")],
    )
    assert stats["marie curie"].df == 2
    assert stats["lyon"].df == 1
    assert stats["marie curie"].labels == {"PERSON": 2}


def test_person_suffix_merges_unambiguous():
    stats = _stats(
        [_ent("Marie Curie", "PERSON")],
        [_ent("Curie", "PERSON")],
        [_ent("Curie", "PERSON")],
    )
    assert build_alias_map(stats) == {"curie": "marie curie"}


def test_person_suffix_ambiguous_stays_split():
    stats = _stats(
        [_ent("Marie Curie", "PERSON")],
        [_ent("Pierre Curie", "PERSON")],
        [_ent("Curie", "PERSON")],
    )
    assert "curie" not in build_alias_map(stats)


def test_acronym_merges_unique_expansion():
    stats = _stats(
        [_ent("United Nations", "ORG")],
        [_ent("UN", "ORG")],
    )
    assert build_alias_map(stats) == {"un": "united nations"}


def test_acronym_ambiguous_expansions_stay_split():
    stats = _stats(
        [_ent("United Nations", "ORG")],
        [_ent("Union Nationale", "ORG")],
        [_ent("UN", "ORG")],
    )
    assert "un" not in build_alias_map(stats)


def test_acronym_needs_uppercase_surface():
    # A lowercase "un" surface is not an acronym candidate.
    stats = _stats([_ent("United Nations", "ORG")], [_ent("un", "ORG")])
    assert build_alias_map(stats) == {}


def test_fuzzy_typo_merges_and_df_picks_canon():
    stats = _stats(
        [_ent("Himmler", "PERSON")],
        [_ent("Himmler", "PERSON")],
        [_ent("Himler", "PERSON")],
    )
    assert build_alias_map(stats) == {"himler": "himmler"}


def test_fuzzy_short_names_never_merge():
    stats = _stats([_ent("Iran", "GPE")], [_ent("Iraq", "GPE")])
    assert build_alias_map(stats) == {}


def test_austria_australia_not_merged():
    stats = _stats([_ent("Austria", "GPE")], [_ent("Australia", "GPE")])
    assert build_alias_map(stats) == {}


def test_directional_contrast_pairs_not_merged():
    # ratio 92.9 + edit distance 2 — passes both numeric gates; the shared-prefix
    # guard is what keeps these distinct.
    stats = _stats([_ent("Western Europe", "LOC")], [_ent("Eastern Europe", "LOC")])
    assert build_alias_map(stats) == {}


def test_dates_and_digits_never_merge():
    stats = _stats(
        [_ent("1942", "DATE")],
        [_ent("1943", "DATE")],
        [_ent("World War I", "EVENT")],
        [_ent("World War II", "EVENT")],
    )
    # "world war i" / "world war ii" contain no digit but ARE distinct — the
    # roman-numeral pair is edit distance 1: guarded by... nothing fuzzy? They pass
    # length >= 5 and have no digits, so this asserts the ratio/edit guards hold.
    amap = build_alias_map(stats)
    assert "1942" not in amap and "1943" not in amap
    assert "world war i" not in amap and "world war ii" not in amap


def test_labels_do_not_cross_blocks():
    # Same-ish strings under incompatible labels never merge.
    stats = _stats([_ent("Jordan", "PERSON")], [_ent("Jordan", "GPE")])
    # single norm "jordan" resolves to ONE label (priority PERSON) — no self-merge
    assert build_alias_map(stats) == {}


def test_gpe_loc_share_a_block():
    stats = _stats(
        [_ent("Auschwitz", "GPE")],
        [_ent("Auschwitz", "GPE")],
        [_ent("Auschwitsz", "LOC")],  # typo tagged LOC
    )
    assert build_alias_map(stats) == {"auschwitsz": "auschwitz"}


def test_canonical_prefers_more_tokens_then_df():
    stats = _stats(
        [_ent("Anne Frank", "PERSON")],
        [_ent("Frank", "PERSON")],
        [_ent("Frank", "PERSON")],
        [_ent("Frank", "PERSON")],
    )
    # "frank" has higher df but "anne frank" has more tokens → canon.
    assert build_alias_map(stats) == {"frank": "anne frank"}


def test_empty_and_none_inputs():
    assert aggregate_entities([]) == {}
    assert aggregate_entities([None, []]) == {}
    assert build_alias_map({}) == {}
