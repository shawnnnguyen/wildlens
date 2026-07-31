"""NODE — Topic redirect fallback (reached when check_relevance says off_topic)."""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage

from wildlens.state import WildlensState

log = logging.getLogger("safari_guide.nodes")


def node_topic_redirect_fallback(state: WildlensState) -> dict:
    """
    Reached when node_check_relevance classifies the message as off_topic.

    Sets final_script so the graph always returns text on this path too.
    Does NOT call the LLM — zero token cost, mirrors
    node_unclear_photo_fallback exactly. Routes straight to the audio gate
    (see graphs.py) rather than through generate_guide_persona, so this
    final_script is never overwritten.
    """
    log.info("▶ NODE  topic_redirect_fallback")
    message = (
        "Ha, that one's outside my wheelhouse! Out here I'm all about the "
        "wildlife — ask me about an animal we've spotted, or point your "
        "camera at something and I'll tell you all about it."
    )
    return {
        "final_script":  message,
        "error_message": "off_topic",
        "chat_history":  [
            HumanMessage(content=state.get("user_message", "")),
            AIMessage(content=message),
        ],
    }
