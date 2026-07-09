"""
Graph-level regression tests:
  - bug #2(b): the unclear_photo_fallback path must never route through
    generate_guide_persona, so its zero-token retake-photo message is never
    overwritten by a fabricated LLM narration.
  - a confident photo identification now routes through summarize_history →
    retrieve_information → generate_guide_persona (safety_check was removed;
    persona no longer narrates a safety warning).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildlens.graphs import build_graph, make_turn_input
from wildlens.rag import _EnsembleRetriever
from wildlens.state import WildlifeIdentification


def _build_test_graph():
    llm_vision = MagicMock()
    llm_text = MagicMock()
    retriever = _EnsembleRetriever(retrievers=[], weights=[])
    graph = build_graph(llm_vision, llm_text, retriever)
    return graph, llm_vision, llm_text


def test_low_confidence_photo_never_calls_persona_llm():
    graph, llm_vision, llm_text = _build_test_graph()

    structured = MagicMock()
    structured.invoke.return_value = MagicMock(
        model_dump=lambda: {
            "species": "unknown", "confidence_score": 0.2,
            "visual_traits": [], "threat_level": "low", "habitat_context": "",
        }
    )
    llm_vision.with_structured_output.return_value = structured

    with patch("wildlens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
        result = graph.invoke(
            make_turn_input(image_path="blurry.jpg"),
            config={"configurable": {"thread_id": "test-fallback"}},
        )

    assert "Kate doesn't guess" in result["final_script"]
    llm_text.invoke.assert_not_called()


def test_confident_photo_routes_through_persona_with_no_safety_alert():
    graph, llm_vision, llm_text = _build_test_graph()

    structured = MagicMock()
    structured.invoke.return_value = WildlifeIdentification(
        species="African Lion (Panthera leo)", confidence_score=0.9,
        visual_traits=["mane"], threat_level="high", habitat_context="savanna",
    )
    llm_vision.with_structured_output.return_value = structured
    llm_text.invoke.return_value = MagicMock(content="Meet the African Lion, genus Panthera, species leo.")

    with patch("wildlens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"), \
         patch.object(_EnsembleRetriever, "retrieve", return_value=[]):
        result = graph.invoke(
            make_turn_input(image_path="lion.jpg"),
            config={"configurable": {"thread_id": "test-confident"}},
        )

    llm_text.invoke.assert_called()  # persona (and summarize/retrieve) now run for photo turns
    assert "SAFETY ALERT" not in result["final_script"]


def test_off_topic_text_turn_never_reaches_persona_llm():
    """The off_topic branch must go straight to the zero-token redirect —
    llm_text.invoke should be called exactly once (the relevance
    classification itself), never a second time for persona generation."""
    graph, llm_vision, llm_text = _build_test_graph()
    llm_text.invoke.return_value = MagicMock(content="OFF_TOPIC")

    result = graph.invoke(
        make_turn_input(user_message="what's the wifi password"),
        config={"configurable": {"thread_id": "test-off-topic"}},
    )

    assert result["error_message"] == "off_topic"
    assert llm_text.invoke.call_count == 1


def test_small_talk_text_turn_skips_retrieval():
    """Small talk should reach persona generation directly, without paying
    for a RAG retrieval call."""
    graph, llm_vision, llm_text = _build_test_graph()
    llm_text.invoke.return_value = MagicMock(content="You're welcome!")

    with patch.object(_EnsembleRetriever, "retrieve") as mock_retrieve:
        graph.invoke(
            make_turn_input(user_message="thanks!"),
            config={"configurable": {"thread_id": "test-small-talk"}},
        )

    mock_retrieve.assert_not_called()
    llm_text.invoke.assert_called()  # persona still generates a reply


def test_on_topic_text_turn_still_retrieves():
    graph, llm_vision, llm_text = _build_test_graph()
    llm_text.invoke.return_value = MagicMock(content="Lions are apex predators.")

    with patch.object(_EnsembleRetriever, "retrieve", return_value=[]) as mock_retrieve:
        result = graph.invoke(
            make_turn_input(user_message="what do predators eat around here?"),
            config={"configurable": {"thread_id": "test-on-topic"}},
        )

    mock_retrieve.assert_called_once()
    assert result["error_message"] != "off_topic"
