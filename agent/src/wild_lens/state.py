"""
State schema and structured-output Pydantic model for the Safari Guide graph.
"""
from __future__ import annotations

import operator
import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ── Configurable thresholds ───────────────────────────────────────────────────
MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.60"))
SUMMARY_THRESHOLD: int = 10  # summarise chat_history when it exceeds this many messages


# ── Graph state ───────────────────────────────────────────────────────────────

class SafariGuideState(TypedDict):
    """
    Central state for the Safari Guide LangGraph.

    Reducer rules
    ─────────────
    chat_history          add_messages  — appends + deduplicates by message id
    identification_history operator.add — list concatenation; never overwritten
    All other fields      last-write-wins (default)

    Caller contract per turn
    ────────────────────────
    Pass only: image_path, user_message, voice_requested, plus the five reset
    fields below.  The MemorySaver checkpointer restores everything else.
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

    # ── Conversation memory (restored by checkpointer between turns) ──────────
    chat_history:           Annotated[list[BaseMessage], add_messages]

    # ── Long-range memory ─────────────────────────────────────────────────────
    identification_history: Annotated[list[dict], operator.add]
    # Every analysed animal is appended here via operator.add so the full session
    # history of species is always available for cross-animal comparison questions.

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
    identification_result:  dict  # keys: species, confidence_score, visual_traits,
                                  #       threat_level, habitat_context, safety_warning*


# ── Structured output schema ──────────────────────────────────────────────────

class WildlifeIdentification(BaseModel):
    """
    Contract enforced by llm.with_structured_output() via Gemini function-calling.
    Literal on threat_level prevents Gemini returning "High!" and silently
    bypassing the == "high" check in node_safety_check.
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
