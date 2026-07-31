"""NODE — Summarise history (long-range memory management)."""
from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from wildlens.nodes._shared import _invoke_with_retry, _strip_synthetic
from wildlens.state import SUMMARY_THRESHOLD, WildlensState

log = logging.getLogger("safari_guide.nodes")


def node_summarize_history(
    state: WildlensState,
    llm: BaseChatModel,
) -> dict:
    """
    Compresses older chat_history turns into conversation_summary when the
    history grows beyond SUMMARY_THRESHOLD messages.

    Only the DELTA since the last summarization call is sent to the LLM —
    tracked via summarized_upto, a persisted (not per-turn-reset) index into
    chat_history — not the entire aged-out prefix every time. This keeps the
    cost of this node bounded instead of growing with conversation length.
    Relies on chat_history only ever growing (add_messages only appends/
    dedups-by-id, never truncates) — if that ever changes, this boundary math
    would need a clamp.

    Returns {} (no-op) when history is short, or when nothing new has aged
    out since the last call — guaranteed safe in LangGraph.
    """
    log.info("▶ NODE  summarize_history")
    history = state.get("chat_history", [])

    if len(history) <= SUMMARY_THRESHOLD:
        log.info("   → %d msgs ≤ threshold (%d). No-op.", len(history), SUMMARY_THRESHOLD)
        return {}

    already  = state.get("summarized_upto", 0)
    boundary = len(history) - 6
    if boundary <= already:
        log.info("   → Nothing new aged out since last summary (boundary=%d, already=%d). No-op.", boundary, already)
        return {}

    # Only the delta since the last summarization call — never the full aged-out prefix
    to_summarise = [msg for msg in _strip_synthetic(history[already:boundary]) if isinstance(msg.content, str)]

    if not to_summarise:
        return {"summarized_upto": boundary}

    prior_summary  = state.get("conversation_summary", "")
    prior_ctx      = f"Prior summary:\n{prior_summary}\n\n" if prior_summary else ""
    messages_text  = "\n".join(
        f"{msg.__class__.__name__}: {msg.content[:400]}"
        for msg in to_summarise
    )

    prompt = HumanMessage(content=(
        f"{prior_ctx}"
        f"New conversation turns to incorporate:\n{messages_text}\n\n"
        "Write a concise factual summary (3–5 sentences) covering: "
        "which animals were discussed and their key facts, any safety alerts given, "
        "and important questions the tourist asked. "
        "This is long-term memory for an ongoing safari conversation."
    ))

    try:
        response = _invoke_with_retry(llm, [prompt])
    except Exception as exc:
        # Non-fatal: skip this turn's summarization rather than failing the whole
        # turn. summarized_upto is left unadvanced, so the same delta is retried
        # on the next call that crosses the threshold.
        log.error("   → summarize_history LLM call failed after retries, skipping: %s", exc)
        return {}

    log.info("   → Conversation summary updated (%d words)", len(response.content.split()))
    return {"conversation_summary": response.content, "summarized_upto": boundary}
