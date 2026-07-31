"""Conditional-edge routing function that follows the analyze_image node."""
from __future__ import annotations

from wildlens.state import MIN_CONFIDENCE, WildlensState


def route_after_analysis(state: WildlensState) -> str:
    """Route based on this turn's confidence score (current_analysis, not the
    last-known-good identification_result — see node_analyze_image)."""
    confidence = state.get("current_analysis", {}).get("confidence_score", 0.0)
    return (
        "unclear_photo_fallback" if confidence < MIN_CONFIDENCE
        else "summarize_history"
    )
