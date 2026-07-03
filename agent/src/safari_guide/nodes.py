"""
LangGraph node implementations for the Safari Guide.

Every node is a pure function: (state, *deps) -> partial_state_dict.
Dependencies (llm, vectorstore) are injected via closures in graphs.py,
not via module-level globals, so each node is independently testable.

Node inventory
──────────────
node_analyze_image         multimodal Gemini vision call → WildlifeIdentification
node_unclear_photo_fallback polite retry prompt when confidence < MIN_CONFIDENCE
node_safety_check          injects safety_warning for high-threat species
node_retrieve_information  FAISS similarity search → retrieved_facts
node_summarize_history     compresses old chat_history → conversation_summary
node_generate_guide_persona Baako persona script generation
node_generate_audio        TTS synthesis (conditional on voice_requested)
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .rag import _EnsembleRetriever
from .state import MIN_CONFIDENCE, SUMMARY_THRESHOLD, SafariGuideState, WildlifeIdentification
from .tts import synthesise_audio

log = logging.getLogger("safari_guide.nodes")

# ── Baako persona (injected first in every generation call) ───────────────────
_BAAKO_SYSTEM = SystemMessage(content=(
    "You are Baako — an energetic, rugged, and deeply passionate African safari guide "
    "with 20 years of experience across the Serengeti, Maasai Mara, and Okavango Delta. "
    "You speak with infectious enthusiasm, weaving scientific accuracy with vivid sensory "
    "descriptions, local Maasai lore, and safety advice delivered as gripping stories. "
    "Your scripts are written for audio delivery: conversational tone, punchy sentences, "
    "no bullet points, no markdown formatting whatsoever. "
    "Aim for 140–220 words (60–90 seconds spoken at a natural pace). "
    "When a SAFETY ALERT is given, open with it in a calm but urgent voice before "
    "transitioning into the wildlife commentary."
))


# ── Image encoding helper ─────────────────────────────────────────────────────

def _to_data_uri(image_path: str) -> str:
    """Return a base64 data URI for *image_path*, or pass through if already a URI."""
    if image_path.startswith("data:"):
        return image_path
    path = Path(image_path)
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
    }
    mime = mime_map.get(path.suffix.lower(), "image/jpeg")
    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode()
    return f"data:{mime};base64,{encoded}"


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Analyse image
# ══════════════════════════════════════════════════════════════════════════════

def node_analyze_image(
    state: SafariGuideState,
    llm: BaseChatModel,
) -> dict:
    """
    Multimodal Gemini call → structured WildlifeIdentification.

    On success: sets identification_result AND appends to identification_history
    (the operator.add reducer accumulates every animal seen this session).

    On failure: sets confidence_score=0.0 so route_after_analysis safely routes
    to the fallback instead of crashing.
    """
    log.info("▶ NODE  analyze_image")
    try:
        structured = llm.with_structured_output(WildlifeIdentification)
        data_uri = _to_data_uri(state["image_path"])

        prompt = HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    "You are an expert wildlife biologist and safari naturalist. "
                    "Examine this image carefully and return a structured identification. "
                    "Set confidence_score below 0.60 for blurry, backlit, or ambiguous images. "
                    "Assign threat_level strictly by the species' inherent danger, not the scene."
                ),
            },
            {"type": "image_url", "image_url": {"url": data_uri}},
        ])

        result: WildlifeIdentification = structured.invoke([prompt])
        log.info(
            "   → %s | conf=%.0f%% | threat=%s",
            result.species, result.confidence_score * 100, result.threat_level,
        )

        ident = result.model_dump()
        return {
            "identification_result":  ident,
            "identification_history": [ident],   # appended via operator.add
            "error_message":          "",
            "chat_history": [
                HumanMessage(content=f"[Photo submitted: {state['image_path']}]"),
                AIMessage(
                    content=(
                        f"Identified **{result.species}** — "
                        f"{result.confidence_score:.0%} confidence, {result.threat_level} threat."
                    )
                ),
            ],
        }

    except Exception as exc:
        log.error("   → analyze_image failed: %s", exc)
        return {
            "identification_result": {"confidence_score": 0.0, "species": "unknown"},
            "error_message":         str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Unclear photo fallback
# ══════════════════════════════════════════════════════════════════════════════

def node_unclear_photo_fallback(state: SafariGuideState) -> dict:
    """
    Reached when confidence_score < MIN_CONFIDENCE.

    Sets final_script so the graph always returns text on this path too.
    Does NOT call the LLM — zero token cost on this low-value path.
    """
    log.info("▶ NODE  unclear_photo_fallback")
    ident      = state.get("identification_result", {})
    confidence = ident.get("confidence_score", 0.0)
    guess      = ident.get("species", "something out in the bush")

    message = (
        f"Ha, I can just about make out what might be {guess} — "
        f"but I'm only {confidence:.0%} confident, and Baako doesn't guess! "
        "Could you try one more shot that's a bit closer, in sharper focus, "
        "and without harsh backlighting? "
        "Once I get a clearer look, I'll have a proper tale for you!"
    )
    return {
        "final_script":  message,
        "error_message": "low_confidence",
        "chat_history":  [AIMessage(content=message)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Safety check
# ══════════════════════════════════════════════════════════════════════════════

def node_safety_check(state: SafariGuideState) -> dict:
    """
    Injects a safety_warning key into identification_result for high-threat species.
    Returns {} for medium/low threat — a confirmed LangGraph no-op.
    Always returns a new dict copy; never mutates state in place.
    """
    log.info("▶ NODE  safety_check")
    ident = state.get("identification_result", {})

    if ident.get("threat_level", "low") != "high":
        log.info("   → Non-high threat; no safety warning injected.")
        return {}

    species = ident.get("species", "this animal")
    warning = (
        f"⚠ SAFETY ALERT — {species.upper()} NEARBY! "
        "Stay inside the vehicle, keep all windows raised, "
        "and make no sudden movements or loud sounds. "
        "Alert your ranger immediately. "
    )
    log.warning("   → HIGH THREAT: %s", species)
    return {"identification_result": {**ident, "safety_warning": warning}}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — RAG retrieval
# ══════════════════════════════════════════════════════════════════════════════

def node_retrieve_information(
    state: SafariGuideState,
    retriever: BaseRetriever,
) -> dict:
    """
    Hybrid BM25 + semantic retrieval for verified guidebook facts.
    Query blends species name + user_message for context-aware retrieval
    on both photo turns and text follow-up turns.
    """
    log.info("▶ NODE  retrieve_information")
    species     = state.get("identification_result", {}).get("species", "")
    common_name = species.split("(")[0].strip()
    follow_up   = state.get("user_message", "")
    query       = f"{common_name} {follow_up}".strip() or "safari wildlife"

    if isinstance(retriever, _EnsembleRetriever):
        docs = retriever.retrieve(query, species=common_name or None)
    else:
        docs = retriever.invoke(query)
    facts = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('species', 'Guidebook')}]\n{d.page_content}"
        for d in docs
    )
    log.info("   → %d docs retrieved | query: '%s'", len(docs), query)
    return {"retrieved_facts": facts}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Summarise history  (long-range memory management)
# ══════════════════════════════════════════════════════════════════════════════

def node_summarize_history(
    state: SafariGuideState,
    llm: BaseChatModel,
) -> dict:
    """
    Compresses older chat_history turns into conversation_summary when the
    history grows beyond SUMMARY_THRESHOLD messages.

    Only the messages older than the most recent 6 are summarised; recent
    turns remain in full. The prior summary is passed as context so each
    call is a rolling update, not a full replay.

    Returns {} (no-op) when history is short — guaranteed safe in LangGraph.
    """
    log.info("▶ NODE  summarize_history")
    history = state.get("chat_history", [])

    if len(history) <= SUMMARY_THRESHOLD:
        log.info("   → %d msgs ≤ threshold (%d). No-op.", len(history), SUMMARY_THRESHOLD)
        return {}

    # Summarise everything except the most recent 6 messages
    to_summarise = [
        msg for msg in history[:-6]
        if isinstance(msg.content, str)
        and not msg.content.startswith("[Photo submitted")
        and not msg.content.startswith("[Conversation memory")
    ]

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

    response = llm.invoke([prompt])
    log.info("   → Conversation summary updated (%d words)", len(response.content.split()))
    return {"conversation_summary": response.content}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Generate guide persona script
# ══════════════════════════════════════════════════════════════════════════════

def node_generate_guide_persona(
    state: SafariGuideState,
    llm: BaseChatModel,
) -> dict:
    """
    Generate Baako's response for the current turn.

    LLM context stack (in order):
      1. Baako system prompt
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
    traits       = ident.get("visual_traits", [])
    safety_alert = ident.get("safety_warning", "")
    facts        = state.get("retrieved_facts", "No additional guidebook facts retrieved.")
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

    # Recent 6 messages, excluding synthetic markers
    recent = [
        msg for msg in history
        if not (isinstance(msg, HumanMessage) and isinstance(msg.content, str) and (
            msg.content.startswith("[Photo submitted")
            or msg.content.startswith("[Conversation memory")
        ))
    ][-6:]
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
        task = HumanMessage(content=(
            f"The tourist is asking: \"{follow_up}\"\n\n"
            f"Relevant guidebook facts:\n{facts}{animals_digest}\n\n"
            "Answer as Baako. If the question refers to a previous animal, "
            "use the session memory and animals list above."
        ))
    else:
        safety_prefix = f"SAFETY ALERT — OPEN WITH THIS:\n{safety_alert}\n\n" if safety_alert else ""
        trait_line    = ", ".join(traits) if traits else "its distinctive features"
        task = HumanMessage(content=(
            f"{safety_prefix}"
            f"You have just spotted a {species}! "
            f"Observable traits: {trait_line}.\n\n"
            f"Verified guidebook facts:\n{facts}{animals_digest}\n\n"
            "Generate a captivating audio tour-guide script as Baako."
        ))

    messages  = [_BAAKO_SYSTEM] + context_msgs + [task]
    response  = llm.invoke(messages)
    script    = response.content
    log.info("   → Script generated (%d words)", len(script.split()))

    return {
        "final_script": script,
        "chat_history": [task, AIMessage(content=script)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 7 — Generate audio  (conditional on voice_requested)
# ══════════════════════════════════════════════════════════════════════════════

def node_generate_audio(state: SafariGuideState) -> dict:
    """
    Thin adapter: reads final_script, writes audio_file_path.
    Only reached when voice_requested=True (enforced by route_audio in graphs.py).
    TTS engine swap (e.g. ElevenLabs) requires changes only in tts.py.
    """
    log.info("▶ NODE  generate_audio")
    script = state.get("final_script", "")
    if not script:
        log.warning("   → No script available for TTS.")
        return {"audio_file_path": ""}

    path = synthesise_audio(script)
    log.info("   → Audio saved: %s", path)
    return {"audio_file_path": path}
