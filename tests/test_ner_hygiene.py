"""NER surface/shape hygiene: the pure cleaning + filtering helpers (no spaCy model).

These pin the fixes for the two observed artifact classes: clitic entities ("n't")
and cross-sentence-boundary spans from lettered enumerations ("World War II. B.").
"""

from flashtopic.chunking import split_list_units
from flashtopic.ner import (
    _acceptable_norm,
    _clean_feed,
    _clean_surface,
    _normalize_entity,
)


# --- shape filter -------------------------------------------------------------

def test_clitics_rejected():
    for clitic in ("n't", "'s", "'d", "'ll", "'re", "'ve", "'m"):
        assert not _acceptable_norm(clitic)


def test_non_word_norms_rejected():
    assert not _acceptable_norm("--")
    assert not _acceptable_norm("''")
    assert not _acceptable_norm("a-")
    assert not _acceptable_norm(".x")


def test_real_norms_accepted():
    assert _acceptable_norm("curie")
    assert _acceptable_norm("world war ii")
    # letter+digit names survive the two-consecutive-letters rule
    assert _acceptable_norm("u2")
    assert _acceptable_norm("3m")
    assert _acceptable_norm("ww2")


def test_digit_only_norms_accepted():
    # DATE entities like "1942" have no letters at all but are legitimate.
    assert _acceptable_norm("1942")
    assert _normalize_entity("1942") == "1942"


# --- cross-sentence-boundary truncation ---------------------------------------

def test_boundary_leak_truncated():
    assert _clean_surface("World War II. B.") == "World War II"


def test_boundary_leak_with_following_words():
    assert _clean_surface("World War II. B. The camp") == "World War II"


def test_title_abbreviations_kept():
    assert _clean_surface("St. Louis") == "St. Louis"
    assert _clean_surface("Mt. Everest") == "Mt. Everest"


def test_middle_initial_kept():
    assert _clean_surface("John F. Kennedy") == "John F. Kennedy"


def test_acronyms_dedotted_before_truncation():
    assert _clean_surface("U.S. Army") == "US Army"


def test_plain_surfaces_untouched():
    assert _clean_surface("Marie Curie") == "Marie Curie"
    assert _clean_surface("the United Nations") == "United Nations"


# --- lettered-enumeration splitting -------------------------------------------

def test_lettered_enum_splits():
    units = split_list_units(
        "A. The war ended in 1945. B. The camps were liberated. C. Trials began."
    )
    assert units == [
        "A. The war ended in 1945.",
        "B. The camps were liberated.",
        "C. Trials began.",
    ]


def test_digit_enum_still_splits():
    units = split_list_units("1. First point. 2. Second point.")
    assert units == ["1. First point.", "2. Second point."]


def test_mid_name_initial_not_split():
    units = split_list_units("He told J. Robert Oppenheimer the news.")
    assert units == ["He told J. Robert Oppenheimer the news."]


# --- feed hygiene: lettered leading markers -----------------------------------

def test_lettered_lead_marker_stripped():
    assert _clean_feed("B. The camp was liberated.") == "The camp was liberated."
    assert _clean_feed("b) The camp was liberated.") == "The camp was liberated."


def test_plain_sentence_lead_untouched():
    assert _clean_feed("I think it ended in 1945.") == "I think it ended in 1945."
