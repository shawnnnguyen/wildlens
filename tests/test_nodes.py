"""
Unit tests for Safari Guide nodes.

Each node is tested with a mocked LLM and/or in-memory FAISS so these tests
run without a GOOGLE_API_KEY and without network access.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from safari_guide.state import MIN_CONFIDENCE, SafariGuideState, WildlifeIdentification
from safari_guide.nodes import (
    node_unclear_photo_fallback,
    node_safety_check,
    node_generate_audio,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_state(**overrides) -> SafariGuideState:
    defaults = SafariGuideState(
        image_path="",
        user_message="",
        voice_requested=False,
        chat_history=[],
        identification_history=[],
        conversation_summary="",
        identification_result={},
        retrieved_facts="",
        final_script="",
        audio_file_path="",
        error_message="",
    )
    defaults.update(overrides)
    return defaults


# ── node_unclear_photo_fallback ───────────────────────────────────────────────

def test_fallback_always_sets_final_script():
    state = _base_state(
        identification_result={"confidence_score": 0.3, "species": "African Lion (Panthera leo)"}
    )
    result = node_unclear_photo_fallback(state)
    assert result["final_script"], "final_script must be non-empty on fallback path"
    assert result["error_message"] == "low_confidence"
    assert len(result["chat_history"]) == 1


def test_fallback_mentions_confidence():
    state = _base_state(
        identification_result={"confidence_score": 0.45, "species": "Zebra"}
    )
    result = node_unclear_photo_fallback(state)
    assert "45%" in result["final_script"]


# ── node_safety_check ─────────────────────────────────────────────────────────

def test_safety_check_noop_for_low_threat():
    state = _base_state(
        identification_result={"species": "Plains Zebra", "threat_level": "low"}
    )
    result = node_safety_check(state)
    assert result == {}, "Must return empty dict (no-op) for non-high threat"


def test_safety_check_injects_warning_for_high_threat():
    state = _base_state(
        identification_result={"species": "African Lion (Panthera leo)", "threat_level": "high"}
    )
    result = node_safety_check(state)
    assert "safety_warning" in result["identification_result"]
    assert "SAFETY ALERT" in result["identification_result"]["safety_warning"]


def test_safety_check_does_not_mutate_state():
    original = {"species": "Hippo", "threat_level": "high"}
    state = _base_state(identification_result=original)
    node_safety_check(state)
    # original dict must not have been mutated
    assert "safety_warning" not in original


# ── node_generate_audio ───────────────────────────────────────────────────────

def test_generate_audio_returns_empty_when_no_script():
    state = _base_state(final_script="")
    result = node_generate_audio(state)
    assert result["audio_file_path"] == ""


def test_generate_audio_calls_synthesise(tmp_path):
    state = _base_state(final_script="The lion roars across the savanna.")
    with patch("safari_guide.nodes.synthesise_audio", return_value=str(tmp_path / "test.mp3")) as mock_tts:
        result = node_generate_audio(state)
    mock_tts.assert_called_once_with("The lion roars across the savanna.")
    assert result["audio_file_path"].endswith("test.mp3")


# ── WildlifeIdentification schema ────────────────────────────────────────────

def test_threat_level_literal_rejects_invalid():
    with pytest.raises(Exception):
        WildlifeIdentification(
            species="Test",
            confidence_score=0.9,
            visual_traits=["big"],
            threat_level="extreme",  # not in Literal["low","medium","high"]
            habitat_context="savanna",
        )


def test_confidence_score_bounds():
    with pytest.raises(Exception):
        WildlifeIdentification(
            species="Test",
            confidence_score=1.5,  # > 1.0
            visual_traits=[],
            threat_level="low",
            habitat_context="forest",
        )
