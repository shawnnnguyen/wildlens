"""
Unit tests for species_lookup.py — the curated species_list.json lookup
used to cross-check threat_level (bug #1) and canonicalize species names
for retrieval filtering (bug #10).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wild_lens.data.species_lookup import (
    canonical_common_name,
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
