"""
Unit tests for species_lookup.py — the curated species_list.json lookup
used to cross-check threat_level (bug #1) and canonicalize species names
for retrieval filtering (bug #10).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildlens.data.species_lookup import (
    canonical_common_name,
    find_mentioned_species,
    ground_truth_threat_level,
    lookup_species,
)


def test_lookup_exact_common_name():
    entry = lookup_species("African Lion")
    assert entry is not None
    assert entry["common_name"] == "African Lion"


def test_lookup_strips_scientific_name():
    entry = lookup_species("African Lion (Panthera leo)")
    assert entry is not None
    assert entry["common_name"] == "African Lion"


def test_lookup_case_and_whitespace_insensitive():
    entry = lookup_species("  african   lion  (Panthera leo)")
    assert entry is not None
    assert entry["common_name"] == "African Lion"


def test_lookup_unknown_species_returns_none():
    assert lookup_species("Unicorn") is None
    assert lookup_species("") is None


def test_canonical_common_name():
    assert canonical_common_name("african lion") == "African Lion"
    assert canonical_common_name("Unicorn") is None


def test_ground_truth_threat_level():
    assert ground_truth_threat_level("African Lion (Panthera leo)") == "high"
    assert ground_truth_threat_level("African Elephant") == "medium"
    assert ground_truth_threat_level("Unicorn") is None


# ── find_mentioned_species ─────────────────────────────────────────────────────

def test_find_mentioned_species_via_head_noun_alias():
    """'African Elephant' never appears verbatim in casual text — must match
    via the derived head-noun alias ('elephant'), not the full common_name."""
    assert find_mentioned_species("what about elephants tho?") == "African Elephant"


def test_find_mentioned_species_word_boundary_not_substring():
    """'Topi' is a real curated species, but must not match inside unrelated
    words like 'topic' — substring matching would defeat the whole filter."""
    assert find_mentioned_species("is this topic allowed?") is None


def test_find_mentioned_species_generic_head_noun_uses_two_word_alias():
    """'Dog' alone (from African Wild Dog) is too generic to word-match —
    must not fire on an unrelated mention of a pet, but must still match the
    fuller phrase 'wild dog'."""
    assert find_mentioned_species("my dog ran away") is None
    assert find_mentioned_species("tell me about the wild dogs") == "African Wild Dog"


def test_find_mentioned_species_no_match_returns_none():
    assert find_mentioned_species("what is the wifi password") is None
    assert find_mentioned_species("") is None


def test_find_mentioned_species_manual_colloquial_aliases():
    """'Hippo'/'rhino'/'croc' don't derive from any common_name's tail word
    (e.g. 'hippo' isn't a substring of 'Hippopotamus') — must be matched via
    the hand-curated _MANUAL_ALIASES table."""
    assert find_mentioned_species("what about hippos?") == "Hippopotamus"
    assert find_mentioned_species("is a rhino dangerous?") == "White Rhinoceros"
    assert find_mentioned_species("tell me about the croc") == "Nile Crocodile"


def test_find_mentioned_species_python_excluded_as_generic():
    """'Python' alone collides with the programming language — must require
    the fuller 'rock python' phrase, like the dog/bird/tree/monitor cases."""
    assert find_mentioned_species("run this python script for me") is None
    assert find_mentioned_species("what about the rock python?") == "African Rock Python"


def test_find_mentioned_species_collision_prefers_session_history():
    """'Gazelle' is shared by two curated species — must prefer whichever was
    most recently identified this session over an arbitrary default."""
    assert find_mentioned_species("what about gazelles", ["Grant's Gazelle"]) == "Grant's Gazelle"
    assert find_mentioned_species(
        "what about gazelles", ["Grant's Gazelle", "Thomson's Gazelle"]
    ) == "Grant's Gazelle"  # most-recent-first order — first entry wins


def test_find_mentioned_species_collision_without_session_history():
    """No session match for an ambiguous alias — falls back to a candidate
    rather than returning None, since the message clearly names *a* species."""
    result = find_mentioned_species("what about gazelles")
    assert result in ("Thomson's Gazelle", "Grant's Gazelle")
