"""
Conditional-edge routing function that gates entry into the generate_audio
node. Shared by three source nodes in graphs.py — unclear_photo_fallback,
topic_redirect_fallback, and generate_guide_persona all route through this
same gate before either reaching generate_audio or ending the turn.
"""
from __future__ import annotations

from langgraph.graph import END

from wildlens.state import WildlensState


def route_audio(state: WildlensState) -> str:
    """Run TTS only when the caller explicitly requests voice output."""
    return "generate_audio" if state.get("voice_requested", False) else END
