"""
Unit tests for the SafariGuideState reducers (Phase 1 hardening: bounded
identification_history).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildlens.state import MAX_IDENTIFICATION_HISTORY, _bounded_identification_history


def test_bounded_identification_history_concatenates_like_operator_add():
    existing = [{"species": "Lion"}]
    new = [{"species": "Zebra"}]
    assert _bounded_identification_history(existing, new) == [{"species": "Lion"}, {"species": "Zebra"}]


def test_bounded_identification_history_caps_at_max():
    existing = [{"species": f"Animal {i}"} for i in range(MAX_IDENTIFICATION_HISTORY)]
    new = [{"species": "Newcomer"}]

    result = _bounded_identification_history(existing, new)

    assert len(result) == MAX_IDENTIFICATION_HISTORY
    assert result[-1] == {"species": "Newcomer"}
    assert result[0] == {"species": "Animal 1"}  # oldest entry ("Animal 0") dropped


def test_bounded_identification_history_never_exceeds_max_even_with_large_batch():
    existing: list[dict] = []
    new = [{"species": f"Animal {i}"} for i in range(MAX_IDENTIFICATION_HISTORY + 5)]

    result = _bounded_identification_history(existing, new)

    assert len(result) == MAX_IDENTIFICATION_HISTORY
    assert result[-1] == {"species": f"Animal {MAX_IDENTIFICATION_HISTORY + 4}"}
