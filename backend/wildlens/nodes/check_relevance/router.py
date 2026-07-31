"""Conditional-edge routing function that follows the check_relevance node."""
from __future__ import annotations

from wildlens.state import WildlensState


def route_after_relevance(state: WildlensState) -> str:
    """
    Route based on node_check_relevance's verdict:
      off_topic  -> a zero-cost templated redirect, never through persona
      small_talk -> straight to persona generation, skipping
                    summarize_history/retrieve_information entirely (a
                    greeting/thanks doesn't need RAG context, and this
                    avoids paying for a retrieval — including a possible
                    Tavily call — on a message with nothing to retrieve)
      on_topic   -> the normal summarise -> retrieve -> persona path
    """
    status = state.get("message_relevance", {}).get("status", "on_topic")
    if status == "off_topic":
        return "topic_redirect_fallback"
    if status == "small_talk":
        return "generate_guide_persona"
    return "summarize_history"
