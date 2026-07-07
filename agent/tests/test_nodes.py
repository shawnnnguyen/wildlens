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

from wildlens.state import MIN_CONFIDENCE, SafariGuideState, WildlifeIdentification
from wildlens.nodes import (
    _embedding_classify_relevance,
    _is_small_talk,
    _is_synthetic_marker,
    _strip_synthetic,
    _to_data_uri,
    _OFF_TOPIC_EXEMPLARS,
    _ON_TOPIC_EXEMPLARS,
    node_analyze_image,
    node_check_relevance,
    node_generate_guide_persona,
    node_retrieve_information,
    node_summarize_history,
    node_topic_redirect_fallback,
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
        message_relevance={},
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
    with patch("wildlens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
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
    with patch("wildlens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
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
    with patch("wildlens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
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
    with patch("wildlens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
        turn1 = node_analyze_image(state, _mock_llm_returning(lion))
    state.update(turn1)

    state["image_path"] = "blurry.jpg"
    with patch("wildlens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
        turn2 = node_analyze_image(state, _mock_llm_returning(blurry))
    state.update(turn2)

    assert state["identification_result"]["species"] == "African Lion (Panthera leo)"
    assert state["current_analysis"]["species"] == "unknown"


# ── node_check_relevance ──────────────────────────────────────────────────────

def _mock_llm_content(text: str) -> MagicMock:
    """A llm.invoke(...) stub whose .content is a real string, not a bare
    MagicMock — needed because MagicMock's default __bool__/.startswith()
    chain is truthy, which would silently make _llm_classify_relevance
    misclassify everything as off_topic if content were left unconfigured."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=text)
    return llm


def test_check_relevance_species_mention_skips_llm():
    llm = MagicMock()
    state = _base_state(user_message="what about elephants tho?")
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "on_topic"
    assert result["message_relevance"]["mentioned_species"] == "African Elephant"
    llm.invoke.assert_not_called()


def test_check_relevance_wildlife_keyword_skips_llm():
    llm = MagicMock()
    state = _base_state(user_message="what do predators eat around here?")
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "on_topic"
    assert result["message_relevance"]["mentioned_species"] is None
    llm.invoke.assert_not_called()


def test_check_relevance_small_talk_skips_llm():
    llm = MagicMock()
    state = _base_state(user_message="thanks so much!")
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "small_talk"
    llm.invoke.assert_not_called()


def test_check_relevance_greeting_plus_real_question_is_not_small_talk():
    """A message containing BOTH a small-talk phrase and a real wildlife
    question must not be misrouted to small_talk (which would skip
    retrieval) — species/keyword matches must win over a mere greeting."""
    llm = MagicMock()
    state = _base_state(user_message="Hi Baako, what do lions eat?")
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "on_topic"
    assert result["message_relevance"]["mentioned_species"] == "African Lion"

    state2 = _base_state(user_message="Thanks! What about elephants?")
    result2 = node_check_relevance(state2, llm)
    assert result2["message_relevance"]["status"] == "on_topic"
    assert result2["message_relevance"]["mentioned_species"] == "African Elephant"


def test_check_relevance_ambiguous_message_uses_llm_off_topic():
    llm = _mock_llm_content("OFF_TOPIC")
    state = _base_state(user_message="what's the wifi password")
    result = node_check_relevance(state, llm)
    assert result["message_relevance"] == {
        "status": "off_topic", "mentioned_species": None, "classification_failed": False,
    }
    llm.invoke.assert_called_once()


def test_check_relevance_ambiguous_message_uses_llm_on_topic():
    llm = _mock_llm_content("ON_TOPIC")
    state = _base_state(user_message="tell me something interesting")
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "on_topic"


def test_check_relevance_llm_error_fails_open_and_flags_failure():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("api down")
    state = _base_state(user_message="what's the wifi password")
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "on_topic"
    assert result["message_relevance"]["classification_failed"] is True


def test_check_relevance_verbose_reply_not_misparsed_as_off_topic():
    """A verbose reply containing the substring 'OFF_TOPIC' must not be
    misclassified — only a response that actually starts with OFF counts."""
    llm = _mock_llm_content("This message is not OFF_TOPIC, it's about wildlife.")
    state = _base_state(user_message="tell me something interesting")
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "on_topic"


def test_check_relevance_species_collision_uses_session_history():
    llm = MagicMock()
    state = _base_state(
        user_message="what about gazelles?",
        identification_history=[{"species": "Grant's Gazelle (Nanger granti)"}],
    )
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["mentioned_species"] == "Grant's Gazelle"


def test_check_relevance_pronoun_followup_with_session_species_is_on_topic():
    """A2: a contextual pronoun follow-up ('can it swim?') with no keyword or
    alias match must be classified on_topic when session_species is threaded
    into the LLM fallback, and the prompt sent to the LLM must mention it."""
    llm = _mock_llm_content("ON_TOPIC")
    state = _base_state(
        user_message="can it swim?",
        identification_history=[{"species": "African Lion (Panthera leo)"}],
    )
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "on_topic"
    llm.invoke.assert_called_once()
    prompt_sent = llm.invoke.call_args[0][0][0].content
    assert "African Lion" in prompt_sent


def test_check_relevance_small_talk_with_session_species_defers_to_llm():
    """A3: 'hey' + a real question ('does it bite?') must not be swallowed
    as small_talk (which would skip retrieval) — _is_small_talk's strict
    whole-message matching means this never matches the phrase set at all,
    so (with no embeddings configured here) it falls through to the
    context-aware LLM fallback instead of the free small-talk heuristic."""
    llm = _mock_llm_content("ON_TOPIC")
    state = _base_state(
        user_message="hey, does it bite?",
        identification_history=[{"species": "Nile Crocodile"}],
    )
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "on_topic"
    llm.invoke.assert_called_once()


def test_check_relevance_bare_pleasantry_with_session_species_still_skips_llm():
    """Regression guard: a bare pleasantry ('thanks so much!') must stay on
    the free small-talk fast path even once an animal is already in play
    this session — identification_history only accumulates and never clears
    mid-session, so a design that gated the fast path on "is a species in
    play" would kill it for nearly every turn after the first
    identification. _is_small_talk's whole-message match (filler words
    stripped) correctly recognizes this as pure small talk regardless."""
    llm = MagicMock()
    state = _base_state(
        user_message="thanks so much!",
        identification_history=[{"species": "Nile Crocodile"}],
    )
    result = node_check_relevance(state, llm)
    assert result["message_relevance"]["status"] == "small_talk"
    llm.invoke.assert_not_called()


# ── _is_small_talk (strict whole-message matching) ─────────────────────────────

def test_is_small_talk_matches_bare_phrases_and_filler_variants():
    assert _is_small_talk("hi") is True
    assert _is_small_talk("Hey!") is True
    assert _is_small_talk("thanks so much!") is True
    assert _is_small_talk("thank you very much") is True
    assert _is_small_talk("good morning!") is True


def test_is_small_talk_rejects_phrase_plus_real_content():
    """The original gap this whole redesign targets: a greeting/thanks
    PREFIXING a real (possibly entirely off-topic) question must not count
    as small talk just because it contains a listed phrase somewhere."""
    assert _is_small_talk("hi, where is my mom?") is False
    assert _is_small_talk("hey, does it bite?") is False
    assert _is_small_talk("hey, what's the weather like?") is False


# ── _embedding_classify_relevance ───────────────────────────────────────────────

class _FakeEmbeddings:
    """Deterministic 2-D embeddings stub standing in for the real local
    sentence-transformer model (see rag/factory.py) — keeps these tests fast
    and offline while still exercising the real cosine-similarity/margin
    logic in _embedding_classify_relevance. All curated exemplar sentences
    map to a fixed on-topic/off-topic cluster vector by default; specific
    query texts are overridden via `vectors`."""

    _ON_TOPIC_VECTOR = [1.0, 0.0]
    _OFF_TOPIC_VECTOR = [0.0, 1.0]

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def _vector_for(self, text: str) -> list[float]:
        if text in self._vectors:
            return self._vectors[text]
        if text in _ON_TOPIC_EXEMPLARS:
            return self._ON_TOPIC_VECTOR
        if text in _OFF_TOPIC_EXEMPLARS:
            return self._OFF_TOPIC_VECTOR
        raise AssertionError(f"no fake vector configured for {text!r}")

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]


def test_embedding_classify_relevance_on_topic():
    embeddings = _FakeEmbeddings({"does it hurt people": [0.9, 0.1]})
    assert _embedding_classify_relevance("does it hurt people", embeddings) == "on_topic"


def test_embedding_classify_relevance_off_topic():
    embeddings = _FakeEmbeddings({"where can I buy souvenirs": [0.1, 0.9]})
    assert _embedding_classify_relevance("where can I buy souvenirs", embeddings) == "off_topic"


def test_embedding_classify_relevance_ambiguous_returns_none():
    """Scores too close together (within _RELEVANCE_MARGIN) must defer to
    the caller's LLM fallback rather than guessing."""
    embeddings = _FakeEmbeddings({"tell me more": [0.5, 0.5]})
    assert _embedding_classify_relevance("tell me more", embeddings) is None


def test_embedding_classify_relevance_error_returns_none():
    embeddings = MagicMock()
    embeddings.embed_query.side_effect = RuntimeError("model not loaded")
    assert _embedding_classify_relevance("does it bite", embeddings) is None


# ── node_check_relevance + embedding tier integration ───────────────────────────

def test_check_relevance_embedding_classifies_common_followup_without_llm():
    """The embedding tier should resolve a common animal-behavior follow-up
    on its own, without ever calling the LLM — this is the concrete case
    that motivated adding this tier: 'does it bite'/'can it swim' are
    extremely common and shouldn't depend on a non-deterministic LLM call."""
    llm = MagicMock()
    embeddings = _FakeEmbeddings({"does it bite": [0.9, 0.1]})
    state = _base_state(user_message="does it bite")
    result = node_check_relevance(state, llm, embeddings)
    assert result["message_relevance"]["status"] == "on_topic"
    llm.invoke.assert_not_called()


def test_check_relevance_greeting_prefixed_off_topic_question_not_small_talk():
    """'hi, where is my mom?' must not be swallowed as small_talk just
    because it contains 'hi' — it falls through to the embedding classifier,
    which (given real off-topic content) correctly says off_topic, without
    ever needing the LLM."""
    llm = MagicMock()
    embeddings = _FakeEmbeddings({"hi, where is my mom?": [0.1, 0.9]})
    state = _base_state(user_message="hi, where is my mom?")
    result = node_check_relevance(state, llm, embeddings)
    assert result["message_relevance"]["status"] == "off_topic"
    llm.invoke.assert_not_called()


def test_check_relevance_embedding_ambiguous_falls_through_to_llm():
    llm = _mock_llm_content("ON_TOPIC")
    embeddings = _FakeEmbeddings({"tell me more": [0.5, 0.5]})
    state = _base_state(user_message="tell me more")
    result = node_check_relevance(state, llm, embeddings)
    assert result["message_relevance"]["status"] == "on_topic"
    llm.invoke.assert_called_once()


def test_check_relevance_no_embeddings_falls_through_to_llm_gracefully():
    """embeddings=None (e.g. the retriever backing this graph has no
    embedding model configured) must skip the embedding tier without
    erroring, not crash node_check_relevance."""
    llm = _mock_llm_content("OFF_TOPIC")
    state = _base_state(user_message="hi, where is my mom?")
    result = node_check_relevance(state, llm, embeddings=None)
    assert result["message_relevance"]["status"] == "off_topic"
    llm.invoke.assert_called_once()


# ── node_topic_redirect_fallback ──────────────────────────────────────────────

def test_topic_redirect_fallback_sets_final_script_and_error():
    state = _base_state(user_message="what's the wifi password")
    result = node_topic_redirect_fallback(state)
    assert result["final_script"]
    assert result["error_message"] == "off_topic"
    assert len(result["chat_history"]) == 2
    assert result["chat_history"][0].content == "what's the wifi password"


# ── node_retrieve_information ────────────────────────────────────────────────

def test_retrieve_information_canonicalizes_species_name():
    """Bug #10: casing/whitespace drift in Gemini's output must be canonicalized
    against species_list.json before being used as the retriever's species filter."""
    from wildlens.rag import _EnsembleRetriever

    retriever = _EnsembleRetriever(retrievers=[], weights=[])
    state = _base_state(identification_result={"species": "african  lion (panthera leo)"})
    with patch.object(_EnsembleRetriever, "retrieve", return_value=[]) as mock_retrieve:
        node_retrieve_information(state, retriever)
    mock_retrieve.assert_called_once()
    _, kwargs = mock_retrieve.call_args
    assert kwargs["species"] == "African Lion"


def test_retrieve_information_prefers_mentioned_species_over_identification_result():
    """Cross-animal follow-up: identification_result is still the lion, but
    this turn's message names the elephant — retrieval must target the
    elephant, not silently keep filtering on the stale identification."""
    from wildlens.rag import _EnsembleRetriever

    retriever = _EnsembleRetriever(retrievers=[], weights=[])
    state = _base_state(
        identification_result={"species": "African Lion (Panthera leo)"},
        message_relevance={"status": "on_topic", "mentioned_species": "African Elephant"},
    )
    with patch.object(_EnsembleRetriever, "retrieve", return_value=[]) as mock_retrieve:
        node_retrieve_information(state, retriever)
    _, kwargs = mock_retrieve.call_args
    assert kwargs["species"] == "African Elephant"


# ── Knowledge Base Enrichment (enqueued from node_retrieve_information) ──────

def test_retrieve_information_enriches_web_docs_only():
    """Web docs returned by retrieval get handed to enrich_async; curated
    guidebook docs (any other source) must never be enriched."""
    from langchain_core.documents import Document
    from wildlens.rag import _EnsembleRetriever

    web_doc = Document(
        page_content="Lions are most active at dawn and dusk.",
        metadata={"source": "web", "url": "http://example.com/lion-facts", "title": "Lion Facts"},
    )
    guidebook_doc = Document(
        page_content="Lions live in prides.",
        metadata={"source": "eol", "species": "African Lion"},
    )

    retriever = _EnsembleRetriever(retrievers=[], weights=[])
    state = _base_state(identification_result={"species": "African Lion (Panthera leo)"})

    with (
        patch.object(_EnsembleRetriever, "retrieve", return_value=[web_doc, guidebook_doc]),
        patch.object(_EnsembleRetriever, "enrich_async") as mock_enrich,
    ):
        node_retrieve_information(state, retriever)

    mock_enrich.assert_called_once()
    _, kwargs = mock_enrich.call_args
    assert kwargs["species"] == "African Lion"
    assert kwargs["content"] == web_doc.page_content
    assert kwargs["source_url"] == "http://example.com/lion-facts"


def test_enqueue_enrichment_gives_each_web_doc_a_distinct_section():
    """Multiple web docs from the same query must not share one section slug —
    upsert_document's delete-then-insert would make each write clobber the last,
    keeping only the final doc instead of all of them."""
    from langchain_core.documents import Document
    from wildlens.nodes import _enqueue_enrichment
    from wildlens.rag import _EnsembleRetriever

    docs = [
        Document(page_content=f"fact {i}", metadata={"source": "web", "url": f"http://x/{i}"})
        for i in range(3)
    ]
    retriever = _EnsembleRetriever(retrievers=[], weights=[])
    with patch.object(_EnsembleRetriever, "enrich_async") as mock_enrich:
        _enqueue_enrichment(retriever, "African Lion", "lion diet", docs)

    sections = [call.kwargs["section"] for call in mock_enrich.call_args_list]
    assert len(sections) == 3
    assert len(set(sections)) == 3


def test_enqueue_enrichment_section_stable_across_calls():
    """Same URL must map to the same section every time it's enriched (sha1,
    not builtin hash()) — PYTHONHASHSEED randomizes hash() per process, which
    would otherwise turn a repeat scrape of the same page into a brand new
    row instead of an idempotent overwrite of the old one."""
    from langchain_core.documents import Document
    from wildlens.nodes import _enqueue_enrichment
    from wildlens.rag import _EnsembleRetriever

    doc = Document(page_content="fact", metadata={"source": "web", "url": "http://x/1"})
    retriever = _EnsembleRetriever(retrievers=[], weights=[])

    with patch.object(_EnsembleRetriever, "enrich_async") as mock_enrich:
        _enqueue_enrichment(retriever, "African Lion", "lion diet", [doc])
        _enqueue_enrichment(retriever, "African Lion", "lion diet", [doc])

    sections = [call.kwargs["section"] for call in mock_enrich.call_args_list]
    assert sections[0] == sections[1]


def test_format_fact_labels_web_enriched_as_web_not_guidebook():
    """A past Tavily result resurfacing via the BM25 rebuild or the web_cache
    Pinecone namespace (source='web_enriched') must still render as Web, never
    Guidebook — the persona prompt is told to trust Guidebook over Web on
    safety conflicts, so mislabeling here would promote unverified scraped
    text to vetted status."""
    from langchain_core.documents import Document
    from wildlens.nodes import _format_fact

    doc = Document(
        page_content="Lions sleep up to 20 hours a day.",
        metadata={"source": "web_enriched", "species": "African Lion", "section": "diet__abc123"},
    )
    formatted = _format_fact(doc)
    assert formatted.startswith("[Source: Web")
    assert "Guidebook" not in formatted.split("]")[0]


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
    with patch("wildlens.nodes._to_data_uri", return_value="data:image/jpeg;base64,xx"):
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
    with patch("wildlens.nodes.synthesise_audio", return_value=str(tmp_path / "test.mp3")) as mock_tts:
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


# ── Retry + graceful degradation on persistent LLM failure (Phase 1 hardening) ──

def test_summarize_history_llm_failure_retries_then_noop():
    history = [AIMessage(content=f"msg {i}") for i in range(20)]
    state = _base_state(chat_history=history, summarized_upto=0)
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("api down")

    result = node_summarize_history(state, llm)

    assert result == {}  # no-op: summarized_upto/conversation_summary untouched
    assert llm.invoke.call_count == 3  # tenacity's stop_after_attempt(3)


def test_persona_llm_failure_retries_then_returns_apology_script():
    state = _base_state(identification_result={"species": "Zebra", "threat_level": "low"})
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("api down")

    result = node_generate_guide_persona(state, llm)

    assert llm.invoke.call_count == 3  # tenacity's stop_after_attempt(3)
    assert result["final_script"]  # contract: always non-empty, even on failure
    assert result["error_message"] == "api down"
    assert len(result["chat_history"]) == 2
