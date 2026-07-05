"""
Graph construction for the Safari Guide.

Single compiled graph with MemorySaver checkpointer.
All conversational turns — photo or text — flow through one graph using
the same thread_id session key.  The checkpointer restores full state
between turns automatically; callers only pass the new inputs per turn.

Graph topology
──────────────
START
  └─ route_entry
        │
        ├─[image_path set]──► analyze_image
        │                         │
        │         ┌───────────────┴────────────────────┐
        │         │ conf < MIN_CONFIDENCE              │ conf ≥ MIN_CONFIDENCE
        │         ▼                                    ▼
        │  unclear_photo_fallback              safety_check
        │         │                                    │
        │    route_audio                     summarize_history
        │   ┌─────┴─────┐                               │
        │   ▼            ▼                    retrieve_information
        │ generate_audio END                            │
        │   │                              generate_guide_persona
        │  END                                           │
        │                                          route_audio
        │                                       ┌────────┴────────┐
        │                                voice_requested        text only
        │                                       ▼                  ▼
        │                                generate_audio            END
        │                                       │
        │                                      END
        │
        └─[user_message set]──► summarize_history
                                       │
                               retrieve_information
                                       │
                               generate_guide_persona
                                       │
                                 route_audio → …
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from .state import MIN_CONFIDENCE, SafariGuideState
from .nodes import (
    node_analyze_image,
    node_unclear_photo_fallback,
    node_safety_check,
    node_retrieve_information,
    node_summarize_history,
    node_generate_guide_persona,
    node_generate_audio,
)


# ── Routing functions ─────────────────────────────────────────────────────────

def route_entry(state: SafariGuideState) -> str:
    """Route to image analysis if a new photo is provided; otherwise go to summarise."""
    if state.get("image_path", "").strip():
        return "analyze_image"
    return "summarize_history"


def route_after_analysis(state: SafariGuideState) -> str:
    """Route based on this turn's confidence score (current_analysis, not the
    last-known-good identification_result — see node_analyze_image)."""
    confidence = state.get("current_analysis", {}).get("confidence_score", 0.0)
    return (
        "unclear_photo_fallback" if confidence < MIN_CONFIDENCE
        else "safety_check"
    )


def route_audio(state: SafariGuideState) -> str:
    """Run TTS only when the caller explicitly requests voice output."""
    return "generate_audio" if state.get("voice_requested", False) else END


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    llm_vision: BaseChatModel,
    llm_text: BaseChatModel,
    retriever: BaseRetriever,
    tracing_enabled: bool = False,
):
    """
    Compile and return the Safari Guide graph with a MemorySaver checkpointer.

    llm_vision — multimodal model (Gemini) used only for node_analyze_image.
    llm_text   — text model (DeepSeek) used for summarise + persona nodes.
    retriever  — hybrid EnsembleRetriever (BM25 + FAISS) from init_rag().
    tracing_enabled — when True, wraps retrieval and TTS with Langfuse
        @observe() spans. Both call plain Python methods (retriever.retrieve(),
        synthesise_audio()) rather than LangChain Runnable.invoke(), so a
        LangChain-callback-based tracer never sees them otherwise.

    Dependencies are injected via closures — each node remains a pure function
    and can be unit-tested with mocked llm / retriever.

    For production persistence swap MemorySaver() with:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string("safari_sessions.db")
    """

    # Bind dependencies without globals
    def _analyze(s):   return node_analyze_image(s, llm_vision)
    def _retrieve(s):  return node_retrieve_information(s, retriever)
    def _summarize(s): return node_summarize_history(s, llm_text)
    def _persona(s):   return node_generate_guide_persona(s, llm_text)
    _audio = node_generate_audio

    if tracing_enabled:
        from langfuse import observe
        _retrieve = observe(name="retrieve_information", as_type="retriever")(_retrieve)
        _audio    = observe(name="generate_audio", as_type="tool")(_audio)

    g = StateGraph(SafariGuideState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("analyze_image",           _analyze)
    g.add_node("unclear_photo_fallback",  node_unclear_photo_fallback)
    g.add_node("safety_check",            node_safety_check)
    g.add_node("summarize_history",       _summarize)
    g.add_node("retrieve_information",    _retrieve)
    g.add_node("generate_guide_persona",  _persona)
    g.add_node("generate_audio",          _audio)

    # ── Entry: photo vs. text turn ────────────────────────────────────────────
    g.add_conditional_edges(
        START,
        route_entry,
        {
            "analyze_image":    "analyze_image",
            "summarize_history": "summarize_history",
        },
    )

    # ── After image analysis: confidence gate ─────────────────────────────────
    g.add_conditional_edges(
        "analyze_image",
        route_after_analysis,
        {
            "unclear_photo_fallback": "unclear_photo_fallback",
            "safety_check":           "safety_check",
        },
    )

    # ── Fallback path: straight to the audio gate — never through persona, so
    # this final_script (a zero-token retake-photo message) is never overwritten
    # by a fabricated LLM narration of a low-confidence guess.
    g.add_conditional_edges(
        "unclear_photo_fallback",
        route_audio,
        {
            "generate_audio": "generate_audio",
            END:               END,
        },
    )

    # ── Happy path: safety → summarise → retrieve → persona ──────────────────
    g.add_edge("safety_check",          "summarize_history")
    g.add_edge("summarize_history",     "retrieve_information")
    g.add_edge("retrieve_information",  "generate_guide_persona")

    # ── Audio gate: conditional TTS ───────────────────────────────────────────
    g.add_conditional_edges(
        "generate_guide_persona",
        route_audio,
        {
            "generate_audio": "generate_audio",
            END:               END,
        },
    )
    g.add_edge("generate_audio", END)

    return g.compile(checkpointer=MemorySaver())


# ── Turn input helper ─────────────────────────────────────────────────────────

def make_turn_input(
    image_path: str = "",
    user_message: str = "",
    voice_requested: bool = False,
) -> dict:
    """
    Build the minimal input dict for one graph invocation.

    Resets per-turn output fields to empty strings so stale values from a
    prior turn (restored by the checkpointer) do not bleed through.
    The checkpointer merges this with the restored session state automatically.
    """
    return {
        "image_path":      image_path,
        "user_message":    user_message,
        "voice_requested": voice_requested,
        # Per-turn resets — caller should always include these
        "final_script":      "",
        "audio_file_path":   "",
        "retrieved_facts":   "",
        "error_message":     "",
        "current_analysis":  {},
    }
