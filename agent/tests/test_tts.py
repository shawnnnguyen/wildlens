"""
Unit tests for TTS runtime-failure fallthrough (Phase 1 hardening).

Previously synthesise_audio() only handled *import-time* unavailability
(neither edge-tts nor gTTS installed) — a *runtime* failure (e.g. a network
drop mid-synthesis) raised uncaught out of the function, which would crash
the whole /chat turn (including the already-generated text response) instead
of degrading to text-only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import wildlens.tts as tts


def test_edge_tts_runtime_failure_falls_through_to_gtts():
    with patch.object(tts, "_EDGE_TTS_AVAILABLE", True), \
         patch.object(tts, "_GTTS_AVAILABLE", True), \
         patch.object(tts, "_edge_tts_coroutine", new=lambda *a, **kw: object()), \
         patch.object(tts, "_run_in_thread", side_effect=RuntimeError("network drop")) as mock_edge, \
         patch.object(tts, "_gTTS") as mock_gtts:
        result = tts.synthesise_audio("hello")

    mock_edge.assert_called_once()
    mock_gtts.assert_called_once()
    assert result != "NO_TTS_ENGINE_INSTALLED"


def test_both_engines_fail_at_runtime_returns_sentinel_instead_of_raising():
    with patch.object(tts, "_EDGE_TTS_AVAILABLE", True), \
         patch.object(tts, "_GTTS_AVAILABLE", True), \
         patch.object(tts, "_edge_tts_coroutine", new=lambda *a, **kw: object()), \
         patch.object(tts, "_run_in_thread", side_effect=RuntimeError("network drop")), \
         patch.object(tts, "_gTTS", side_effect=RuntimeError("also down")):
        result = tts.synthesise_audio("hello")

    assert result == "NO_TTS_ENGINE_INSTALLED"


def test_gtts_runtime_failure_returns_sentinel_when_edge_unavailable():
    with patch.object(tts, "_EDGE_TTS_AVAILABLE", False), \
         patch.object(tts, "_GTTS_AVAILABLE", True), \
         patch.object(tts, "_gTTS", side_effect=RuntimeError("network drop")):
        result = tts.synthesise_audio("hello")

    assert result == "NO_TTS_ENGINE_INSTALLED"
