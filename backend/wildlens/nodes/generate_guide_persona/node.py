"""NODE — Generate guide persona script (Kate persona)."""
from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from wildlens.nodes._shared import _invoke_with_retry, _is_truncated, _strip_synthetic, wrap_untrusted
from wildlens.nodes.generate_guide_persona.prompts import (
    _KATE_SYSTEM,
    _PERSONA_FOLLOWUP_TASK_TEMPLATE,
    _PERSONA_INTRO_TASK_TEMPLATE,
)
from wildlens.state import WildlensState

log = logging.getLogger("safari_guide.nodes")


def node_generate_guide_persona(
    state: WildlensState,
    llm: BaseChatModel,
) -> dict:
    """
    Generate Kate's response for the current turn.

    LLM context stack (in order):
      1. Kate system prompt
      2. conversation_summary block — compressed long-range memory
      3. Last 6 chat_history messages — recent turn context
      4. identification_history digest — all animals seen this session
      5. Current task message (photo intro or follow-up answer)

    Always writes final_script — this is the contract that guarantees
    every turn returns text regardless of which path reached this node.
    """
    log.info("▶ NODE  generate_guide_persona")

    ident        = state.get("identification_result", {})
    species      = ident.get("species", "this remarkable creature")
    genus        = ident.get("genus", "")
    species_epithet = ident.get("species_epithet", "")
    traits       = ident.get("visual_traits", [])
    # `.get(key, default)` would never fall back here — retrieve_information always
    # sets the key, even to "" when nothing was found — so use `or` instead.
    facts        = state.get("retrieved_facts") or "No additional guidebook facts retrieved."
    follow_up    = state.get("user_message", "")
    summary      = state.get("conversation_summary", "")
    history      = state.get("chat_history", [])
    id_history   = state.get("identification_history", [])

    # ── Build context messages ────────────────────────────────────────────────
    context_msgs = []

    if summary:
        context_msgs.append(HumanMessage(
            content=f"[Conversation memory — animals and facts from earlier in this tour:\n{summary}]"
        ))

    # Recent 6 messages, excluding synthetic markers. Slice a bounded raw tail
    # BEFORE filtering rather than scanning the entire history every turn
    # (compounds with node_summarize_history on long sessions) — a confident
    # photo turn appends at most 1 marker per 4 raw messages (marker +
    # identified-AIMessage + persona's task + script; a low-confidence photo
    # turn appends 1 non-marker message and no persona call at all — see
    # node_analyze_image / graphs.py's fallback routing), so this 25%
    # worst-case density means _RECENT_RAW_WINDOW messages are always enough
    # to yield >= 6 survivors.
    _RECENT_RAW_WINDOW = 12
    tail = history[-_RECENT_RAW_WINDOW:] if len(history) > _RECENT_RAW_WINDOW else history
    recent = _strip_synthetic(tail)[-6:]
    context_msgs.extend(recent)

    # ── Animals seen this session (for cross-animal questions) ────────────────
    animals_digest = ""
    if len(id_history) > 1:
        lines = ", ".join(
            f"{h.get('species', 'Unknown')} ({h.get('threat_level', '?')} threat)"
            for h in id_history
        )
        animals_digest = f"\n\nAnimals identified this session: {lines}"

    # ── Build the task message for this specific turn ─────────────────────────
    if follow_up:
        task = HumanMessage(content=_PERSONA_FOLLOWUP_TASK_TEMPLATE.format(
            follow_up=wrap_untrusted(follow_up), facts=facts, animals_digest=animals_digest,
        ))
    else:
        trait_line  = ", ".join(traits) if traits else "its distinctive features"
        binomial_line = (
            f"Genus: {genus}. Species: {species_epithet}.\n"
            if genus and species_epithet else ""
        )
        task = HumanMessage(content=_PERSONA_INTRO_TASK_TEMPLATE.format(
            species=species, binomial_line=binomial_line, trait_line=trait_line,
            facts=facts, animals_digest=animals_digest,
        ))

    messages = [_KATE_SYSTEM] + context_msgs + [task]
    try:
        response = _invoke_with_retry(llm, messages)
    except Exception as exc:
        # final_script must always be set (see docstring) — degrade to a fixed,
        # zero-token apology rather than letting this bubble to a generic 500.
        log.error("   → generate_guide_persona LLM call failed after retries: %s", exc)
        script = (
            "Ah, my words got lost somewhere out on the savanna — could you ask "
            "me that again? I want to make sure I get it right for you."
        )
        return {
            "final_script":  script,
            "error_message": str(exc),
            "chat_history":  [task, AIMessage(content=script)],
        }

    script = response.content
    if _is_truncated(response):
        # Unlike analyze_image's structured JSON, a truncated script is still
        # partially usable text — degrading to the generic apology would throw
        # away a mostly-good response, so this only logs for visibility rather
        # than replacing final_script or setting error_message.
        log.warning(
            "   → generate_guide_persona response truncated by token limit (finish_reason=%s)",
            response.response_metadata.get("finish_reason"),
        )
    log.info("   → Script generated (%d words)", len(script.split()))

    return {
        "final_script": script,
        "chat_history": [task, AIMessage(content=script)],
    }
