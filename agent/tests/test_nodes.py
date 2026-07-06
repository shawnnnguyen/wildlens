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

from langchain_core.messages import AIMessage, HumanMessage

from wild_lens.state import MIN_CONFIDENCE, SafariGuideState, WildlifeIdentification
from wild_lens.nodes import (
    _is_synthetic_marker,
    _strip_synthetic,
    _to_data_uri,
    node_analyze_image,
    node_generate_guide_persona,
    node_retrieve_information,
    node_summarize_history,
    node_unclear_photo_fallback,
    node_generate_audio,
    parse_binomial,
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
        summarized_upto=0,
        identification_result={},
        current_analysis={},
        retrieved_facts="",
        final_script="",
        audio_file_path="",
        error_message="",
    )
    defaults.update(overrides)
    return defaults


# ── node_analyze_image ────────────────────────────────────────────────────────

def _mock_llm_returning(identification: WildlifeIdentification) -> MagicMock:
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = identification
    llm.with_structured_output.return_value = structured
    return llm


def test_analyze_image_escalates_threat_level_from_curated_data():
    """Gemini under-calls a curated-'high' species as 'low' — must be escalated,
    and identification_history must carry the SAME escalated value (bug #1)."""
    ident = WildlifeIdentification(
        species="African Lion (Panthera leo)",
        confidence_score=0.9,
        visual_traits=["mane"],
        threat_level="low",
        habitat_context="savanna",
    )
    llm = _mock_llm_returning(ident)
    state = _base_state(image_path="lion.jpg")
    with patch("wild_lens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
        result = node_analyze_image(state, llm)
    assert result["identification_result"]["threat_level"] == "high"
    assert result["identification_history"][0]["threat_level"] == "high"


def test_analyze_image_does_not_downgrade_high_call():
    """Gemini says 'high' for a curated-'medium' species — must NOT be downgraded."""
    ident = WildlifeIdentification(
        species="African Elephant (Loxodonta africana)",
        confidence_score=0.9,
        visual_traits=["tusks"],
        threat_level="high",
        habitat_context="savanna",
    )
    llm = _mock_llm_returning(ident)
    state = _base_state(image_path="elephant.jpg")
    with patch("wild_lens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
        result = node_analyze_image(state, llm)
    assert result["identification_result"]["threat_level"] == "high"


def test_analyze_image_low_confidence_does_not_set_identification_result():
    """Bug #2: a low-confidence analysis must not clobber identification_result,
    but identification_history keeps accumulating unconditionally as before."""
    ident = WildlifeIdentification(
        species="Something Blurry",
        confidence_score=0.2,
        visual_traits=[],
        threat_level="low",
        habitat_context="unknown",
    )
    llm = _mock_llm_returning(ident)
    state = _base_state(image_path="blurry.jpg")
    with patch("wild_lens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
        result = node_analyze_image(state, llm)
    assert "identification_result" not in result
    assert "chat_history" not in result
    assert result["current_analysis"]["species"] == "Something Blurry"
    assert result["identification_history"] == [result["current_analysis"]]


def test_analyze_image_exception_sets_current_analysis_only():
    llm = MagicMock()
    llm.with_structured_output.side_effect = RuntimeError("boom")
    state = _base_state(image_path="broken.jpg")
    result = node_analyze_image(state, llm)
    assert "identification_result" not in result
    assert "identification_history" not in result
    assert result["current_analysis"] == {"confidence_score": 0.0, "species": "unknown"}
    assert result["error_message"] == "boom"


def test_identification_result_survives_a_later_blurry_photo():
    """Regression for bug #2: a confident lion identification, then a blurry
    follow-up photo — identification_result must still be the lion afterward."""
    lion = WildlifeIdentification(
        species="African Lion (Panthera leo)", confidence_score=0.9,
        visual_traits=["mane"], threat_level="high", habitat_context="savanna",
    )
    blurry = WildlifeIdentification(
        species="unknown", confidence_score=0.1,
        visual_traits=[], threat_level="low", habitat_context="unknown",
    )

    state = _base_state(image_path="lion.jpg")
    with patch("wild_lens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
        turn1 = node_analyze_image(state, _mock_llm_returning(lion))
    state.update(turn1)

    state["image_path"] = "blurry.jpg"
    with patch("wild_lens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
        turn2 = node_analyze_image(state, _mock_llm_returning(blurry))
    state.update(turn2)

    assert state["identification_result"]["species"] == "African Lion (Panthera leo)"
    assert state["current_analysis"]["species"] == "unknown"


# ── node_retrieve_information ────────────────────────────────────────────────

def test_retrieve_information_canonicalizes_species_name():
    """Bug #10: casing/whitespace drift in Gemini's output must be canonicalized
    against species_list.json before being used as the retriever's species filter."""
    from wild_lens.rag import _EnsembleRetriever

    retriever = _EnsembleRetriever(retrievers=[], weights=[])
    state = _base_state(identification_result={"species": "african  lion (panthera leo)"})
    with patch.object(_EnsembleRetriever, "retrieve", return_value=[]) as mock_retrieve:
        node_retrieve_information(state, retriever)
    mock_retrieve.assert_called_once()
    _, kwargs = mock_retrieve.call_args
    assert kwargs["species"] == "African Lion"


# ── _to_data_uri ──────────────────────────────────────────────────────────────

def test_to_data_uri_rejects_bad_extension(tmp_path):
    bad = tmp_path / "file.txt"
    bad.write_text("not an image")
    with pytest.raises(ValueError):
        _to_data_uri(str(bad))


def test_to_data_uri_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _to_data_uri(str(tmp_path / "missing.jpg"))


def test_to_data_uri_rejects_oversized_file(tmp_path):
    big = tmp_path / "big.jpg"
    big.write_bytes(b"0" * (10 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        _to_data_uri(str(big))


def test_to_data_uri_encodes_valid_image(tmp_path):
    small = tmp_path / "small.jpg"
    small.write_bytes(b"fake-jpeg-bytes")
    uri = _to_data_uri(str(small))
    assert uri.startswith("data:image/jpeg;base64,")


def test_to_data_uri_passes_through_existing_data_uri():
    uri = "data:image/png;base64,abc123"
    assert _to_data_uri(uri) == uri


# ── node_unclear_photo_fallback ───────────────────────────────────────────────

def test_fallback_always_sets_final_script():
    state = _base_state(
        current_analysis={"confidence_score": 0.3, "species": "African Lion (Panthera leo)"}
    )
    result = node_unclear_photo_fallback(state)
    assert result["final_script"], "final_script must be non-empty on fallback path"
    assert result["error_message"] == "low_confidence"
    assert len(result["chat_history"]) == 1


def test_fallback_mentions_confidence():
    state = _base_state(
        current_analysis={"confidence_score": 0.45, "species": "Zebra"}
    )
    result = node_unclear_photo_fallback(state)
    assert "45%" in result["final_script"]


# ── parse_binomial ────────────────────────────────────────────────────────────

def test_parse_binomial_extracts_genus_and_epithet():
    assert parse_binomial("African Lion (Panthera leo)") == ("Panthera", "leo")


def test_parse_binomial_handles_missing_parentheses():
    assert parse_binomial("unknown") == ("", "")


def test_parse_binomial_handles_empty_string():
    assert parse_binomial("") == ("", "")


def test_analyze_image_sets_genus_and_species_epithet():
    ident = WildlifeIdentification(
        species="African Lion (Panthera leo)", confidence_score=0.9,
        visual_traits=["mane"], threat_level="high", habitat_context="savanna",
    )
    llm = _mock_llm_returning(ident)
    state = _base_state(image_path="lion.jpg")
    with patch("wild_lens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
        result = node_analyze_image(state, llm)
    assert result["identification_result"]["genus"] == "Panthera"
    assert result["identification_result"]["species_epithet"] == "leo"


# ── node_generate_audio ───────────────────────────────────────────────────────

def test_generate_audio_returns_empty_when_no_script():
    state = _base_state(final_script="")
    result = node_generate_audio(state)
    assert result["audio_file_path"] == ""


def test_generate_audio_calls_synthesise(tmp_path):
    state = _base_state(final_script="The lion roars across the savanna.")
    with patch("wild_lens.nodes.synthesise_audio", return_value=str(tmp_path / "test.mp3")) as mock_tts:
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


# ── _strip_synthetic / _is_synthetic_marker (bug #5) ─────────────────────────

def test_strip_synthetic_drops_photo_marker_only():
    marker  = HumanMessage(content="[Photo submitted: lion.jpg]")
    memory  = HumanMessage(content="[Conversation memory ...]")  # never actually in chat_history
    keeper  = AIMessage(content="A lion was spotted.")
    assert _is_synthetic_marker(marker) is True
    assert _is_synthetic_marker(memory) is False  # documents: this shape is never filtered here
    result = _strip_synthetic([marker, memory, keeper])
    assert result == [memory, keeper]


# ── node_generate_guide_persona: bounded tail-slice (bug #7) ─────────────────

def test_persona_recent_messages_bounded_over_consecutive_low_confidence_turns():
    """
    Worst-case marker density scenario: many consecutive low-confidence photo
    turns append 1 plain (non-marker) AIMessage each and never call persona,
    then a final confident photo turn appends marker + identified-AIMessage.
    The bounded 12-message tail must still surface the same last-6 kept
    messages a full-history scan would have produced.
    """
    history = []
    for i in range(10):
        history.append(AIMessage(content=f"low confidence guess #{i}"))
    history.append(HumanMessage(content="[Photo submitted: lion.jpg]"))
    history.append(AIMessage(content="Identified **African Lion** — 90% confidence, high threat."))

    state = _base_state(
        chat_history=history,
        identification_result={"species": "African Lion (Panthera leo)", "threat_level": "high"},
    )
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="A roaring tale of the savanna.")
    node_generate_guide_persona(state, llm)

    sent_messages = llm.invoke.call_args[0][0]
    # Full-scan-equivalent: strip markers from the whole history, take last 6
    expected_recent = _strip_synthetic(history)[-6:]
    # context_msgs = [] (no summary) + recent + [task]; the persona system
    # message is always first, the task message always last.
    assert sent_messages[1:1 + len(expected_recent)] == expected_recent


# ── node_generate_guide_persona: photo-turn task content ─────────────────────

def test_persona_photo_turn_has_no_safety_alert_text():
    """Safety warnings are no longer narrated in the response (dropped per
    product decision — camera-phase warnings are handled by the frontend)."""
    state = _base_state(
        identification_result={
            "species": "African Lion (Panthera leo)", "threat_level": "high",
            "genus": "Panthera", "species_epithet": "leo",
        },
    )
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="A lion at rest.")
    node_generate_guide_persona(state, llm)

    task_message = llm.invoke.call_args[0][0][-1]
    assert "SAFETY ALERT" not in task_message.content


def test_persona_photo_turn_instructs_genus_species_diet_circadian():
    state = _base_state(
        identification_result={
            "species": "African Lion (Panthera leo)", "threat_level": "high",
            "genus": "Panthera", "species_epithet": "leo",
        },
    )
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="A lion at rest.")
    node_generate_guide_persona(state, llm)

    task_message = llm.invoke.call_args[0][0][-1]
    content = task_message.content
    assert "Genus: Panthera" in content
    assert "Species: leo" in content
    assert "circadian rhythm" in content
    assert "diet" in content
    assert "apologize" in content  # graceful-missing-data instruction present


def test_persona_facts_fallback_fires_on_empty_string():
    """retrieve_information always sets retrieved_facts (even to ""), so the
    fallback text must be reachable via `or`, not `.get(key, default)`."""
    state = _base_state(
        identification_result={"species": "Zebra", "threat_level": "low"},
        retrieved_facts="",
    )
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="A zebra grazes.")
    node_generate_guide_persona(state, llm)

    task_message = llm.invoke.call_args[0][0][-1]
    assert "No additional guidebook facts retrieved." in task_message.content


# ── node_summarize_history: incremental summarized_upto (bug #3) ────────────

def test_summarize_history_sends_only_delta_since_last_boundary():
    history = [AIMessage(content=f"msg {i}") for i in range(20)]
    state = _base_state(chat_history=history, summarized_upto=12, conversation_summary="prior digest")
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="updated summary")

    result = node_summarize_history(state, llm)

    sent_prompt = llm.invoke.call_args[0][0][0].content
    assert "msg 12" in sent_prompt
    assert "msg 13" in sent_prompt
    assert "msg 11" not in sent_prompt   # already summarized, must not be resent
    assert "msg 14" not in sent_prompt   # still within the last-6 sliding window
    assert result["summarized_upto"] == 14
    assert result["conversation_summary"] == "updated summary"


def test_summarize_history_noop_when_nothing_new_aged_out():
    history = [AIMessage(content=f"msg {i}") for i in range(20)]
    state = _base_state(chat_history=history, summarized_upto=14)  # boundary is already 14
    llm = MagicMock()
    result = node_summarize_history(state, llm)
    assert result == {}
    llm.invoke.assert_not_called()


def test_summarize_history_first_call_matches_full_resend_behavior():
    """First-ever call (summarized_upto=0) must reproduce today's full-resend
    behavior exactly — no regression on first fire."""
    history = [AIMessage(content=f"msg {i}") for i in range(20)]
    state = _base_state(chat_history=history, summarized_upto=0)
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="summary")
    result = node_summarize_history(state, llm)
    sent_prompt = llm.invoke.call_args[0][0][0].content
    for i in range(14):
        assert f"msg {i}" in sent_prompt
    assert result["summarized_upto"] == 14
