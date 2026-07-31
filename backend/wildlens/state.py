"""
State schema and structured-output Pydantic model for the WildLens graph.
"""
from __future__ import annotations

import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ── Configurable thresholds ───────────────────────────────────────────────────
MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.60"))
SUMMARY_THRESHOLD: int = 10  # summarise chat_history when it exceeds this many messages
MAX_IDENTIFICATION_HISTORY: int = 15  # keep only the most recent N identified animals per session


def _bounded_identification_history(existing: list[dict], new: list[dict]) -> list[dict]:
    """
    Reducer for identification_history: concatenates like operator.add, then
    keeps only the most recent MAX_IDENTIFICATION_HISTORY entries so a long
    session (many photos) can't grow this field — and the persona prompt's
    animals-seen-this-session digest built from it — without bound.
    """
    return (existing + new)[-MAX_IDENTIFICATION_HISTORY:]


# ── Graph state ───────────────────────────────────────────────────────────────

class WildlensState(TypedDict):
    """
    Central state for the WildLens LangGraph.

    Reducer rules
    ─────────────
    chat_history           add_messages                 — appends + deduplicates by message id
    identification_history _bounded_identification_history — appends, keeps last MAX_IDENTIFICATION_HISTORY
    All other fields       last-write-wins (default)

    Caller contract per turn
    ────────────────────────
    Pass only: image_path, user_message, voice_requested, plus the five reset
    fields below.  The checkpointer (MemorySaver or SqliteSaver — see
    graphs.py) restores everything else.
    """

    # ── Caller inputs (set fresh every invoke) ────────────────────────────────
    image_path:             str   # file path or "data:<mime>;base64,…"; "" on text turns
    user_message:           str   # tourist's question; "" on photo turns
    voice_requested:        bool  # True → TTS node runs; False → skip, text-only

    # ── Per-turn resets (caller passes "" / "" to clear stale values) ─────────
    final_script:           str   # reset to "" each turn; always written before END
    audio_file_path:        str   # reset to "" each turn; written only if voice_requested
    retrieved_facts:        str   # reset to "" each turn; written by retrieve node
    error_message:          str   # reset to "" each turn; written on fallback/error
    current_analysis:       dict  # reset to {} each turn; this turn's raw analysis result
                                  # (success or error stub), read by route_after_analysis and
                                  # node_unclear_photo_fallback — never identification_result,
                                  # so a low-confidence/failed photo can't clobber the last
                                  # confidently-identified animal.
    message_relevance:      dict  # reset to {} each turn; written by node_check_relevance for
                                  # text turns only. Keys: "status" ("on_topic" | "small_talk" |
                                  # "off_topic"), "mentioned_species" (canonical common_name if
                                  # the message names a specific animal, overriding
                                  # identification_result's species for this turn's retrieval —
                                  # see node_retrieve_information), "classification_failed"
                                  # (True if the LLM fallback layer errored and this turn
                                  # defaulted open to "on_topic" — surfaced so a persistently
                                  # broken classifier is observable, not silently no-op).

    # ── Conversation memory (restored by checkpointer between turns) ──────────
    chat_history:           Annotated[list[BaseMessage], add_messages]

    # ── Long-range memory ─────────────────────────────────────────────────────
    identification_history: Annotated[list[dict], _bounded_identification_history]
    # Every analysed animal is appended here so the recent session history of
    # species is available for cross-animal comparison questions — bounded to
    # the last MAX_IDENTIFICATION_HISTORY entries so an unusually long session
    # (many photos) can't grow this field, or the persona prompt digest built
    # from it, without limit.

    conversation_summary:   str
    # LLM-compressed digest of older chat_history turns (beyond the 6-message
    # sliding window).  Injected into the persona prompt for long-range recall
    # without growing the LLM context window unboundedly.

    summarized_upto:        int
    # Index boundary into chat_history already folded into conversation_summary.
    # NOT a per-turn reset — must persist across turns like conversation_summary
    # itself. node_summarize_history only sends the delta since this boundary to
    # the LLM, keeping its cost bounded instead of growing with session length.

    # ── Current-animal pipeline data (overwritten each photo turn) ───────────
    identification_result:  dict  # keys: species, genus, species_epithet, confidence_score,
                                  #       visual_traits, threat_level, habitat_context


# ── Structured output schema ──────────────────────────────────────────────────

class WildlifeIdentification(BaseModel):
    """
    Contract enforced by llm.with_structured_output() via Gemini function-calling.
    Literal on threat_level prevents Gemini returning "High!" and silently
    bypassing the curated-ground-truth escalation check in node_analyze_image.
    """

    species: str = Field(
        description="Common name followed by scientific name in parentheses, "
                    "e.g. 'African Lion (Panthera leo)'"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Identification confidence. Set < 0.60 for blurry, backlit, or ambiguous images."
    )
    visual_traits: list[str] = Field(
        description="Key visible features: colour, markings, size, posture, anatomy."
    )
    threat_level: Literal["low", "medium", "high"] = Field(
        description=(
            "'high' for apex predators / acutely dangerous animals (lion, leopard, hippo, croc). "
            "'medium' for strong defensive behaviour (elephant, rhino, baboon). "
            "'low' for generally safe species (zebra, giraffe, antelope, birds, flora)."
        )
    )
    habitat_context: str = Field(
        description="One sentence on the species' typical habitat or ecosystem."
    )
