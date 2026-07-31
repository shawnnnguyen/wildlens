"""
Cross-node helpers shared by more than one node module — kept here instead of
duplicated in each node's node.py.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _invoke_with_retry(llm: Runnable, messages: list):
    """
    Shared retry policy for every plain .invoke() call against Gemini across
    nodes: analyze_image's structured.invoke() (a Runnable, same interface),
    check_relevance's _llm_classify_relevance's llm.invoke(),
    summarize_history's llm.invoke(), and generate_guide_persona's
    llm.invoke(). A transient API blip (timeout, rate limit) gets retried
    here instead of immediately tripping the caller's try/except and failing
    the turn (or, for the relevance classifier, silently failing open).
    """
    return llm.invoke(messages)


def _is_synthetic_marker(msg) -> bool:
    """
    True for the "[Photo submitted ...]" HumanMessage node_analyze_image injects
    into chat_history as a lightweight marker of a photo turn.

    Not to be confused with the "[Conversation memory ...]" context message
    node_generate_guide_persona builds inline for direct LLM context only —
    that one is never appended to chat_history, so no marker check for it is
    needed (or possible to trigger) here.
    """
    return (
        isinstance(msg, HumanMessage)
        and isinstance(msg.content, str)
        and msg.content.startswith("[Photo submitted")
    )


def _strip_synthetic(messages: list) -> list:
    return [m for m in messages if not _is_synthetic_marker(m)]
