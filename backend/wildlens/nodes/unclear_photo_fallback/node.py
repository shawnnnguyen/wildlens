"""NODE — Unclear photo fallback (reached when confidence_score < MIN_CONFIDENCE)."""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from wildlens.state import WildlensState

log = logging.getLogger("safari_guide.nodes")


def node_unclear_photo_fallback(state: WildlensState) -> dict:
    """
    Reached when confidence_score < MIN_CONFIDENCE.

    Sets final_script so the graph always returns text on this path too.
    Does NOT call the LLM — zero token cost on this low-value path.
    Routes straight to the audio gate (see graphs.py) rather than through
    generate_guide_persona, so this final_script is never overwritten.
    """
    log.info("▶ NODE  unclear_photo_fallback")
    ident      = state.get("current_analysis", {})
    confidence = ident.get("confidence_score", 0.0)
    guess      = ident.get("species", "something out in the bush")

    message = (
        f"Ha, I can just about make out what might be {guess} — "
        f"but I'm only {confidence:.0%} confident, and Kate doesn't guess! "
        "Could you try one more shot that's a bit closer, in sharper focus, "
        "and without harsh backlighting? "
        "Once I get a clearer look, I'll have a proper tale for you!"
    )
    return {
        "final_script":  message,
        # node_analyze_image sets error_message="truncated_response" (not a
        # confidence issue at all — confidence_score=0.0 is just the stub
        # value, which routes here) and chat.py needs that exact sentinel to
        # surface a 502 instead of a silent low-confidence fallback. Only
        # default to "low_confidence" when analyze_image didn't already set
        # a more specific error.
        "error_message": state.get("error_message") or "low_confidence",
        "chat_history":  [AIMessage(content=message)],
    }
