"""
LangGraph node implementations for the Safari Guide.

Every node is a pure function: (state, *deps) -> partial_state_dict.
Dependencies (llm, vectorstore) are injected via closures in graphs.py,
not via module-level globals, so each node is independently testable.

Node inventory
──────────────
node_analyze_image         multimodal Gemini vision call → WildlifeIdentification
node_unclear_photo_fallback polite retry prompt when confidence < MIN_CONFIDENCE
node_retrieve_information  hybrid RAG search → retrieved_facts
node_summarize_history     compresses old chat_history → conversation_summary
node_generate_guide_persona Baako persona script generation
node_generate_audio        TTS synthesis (conditional on voice_requested)
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .data.species_lookup import canonical_common_name, ground_truth_threat_level
from .rag import _EnsembleRetriever
from .state import MIN_CONFIDENCE, SUMMARY_THRESHOLD, SafariGuideState, WildlifeIdentification
from .tts import synthesise_audio

log = logging.getLogger("safari_guide.nodes")

# Ordering used to escalate (never downgrade) Gemini's live threat_level call
# against species_list.json's curated ground truth — see node_analyze_image.
_THREAT_RANK = {"low": 0, "medium": 1, "high": 2}

# ── Baako persona (injected first in every generation call) ───────────────────
_BAAKO_SYSTEM = SystemMessage(content=(
    "You are Baako — a knowledgeable and enthusiastic African safari guide with 20 years "
    "of experience across the Serengeti, Maasai Mara, and Okavango Delta. "
    "You speak with genuine warmth and respect for the wildlife you describe, grounding "
    "your enthusiasm in scientific accuracy rather than theatrics. "
    "Your scripts are written for audio delivery: conversational tone, punchy sentences, "
    "no bullet points, no markdown formatting whatsoever. "
    "Aim for 140–220 words (60–90 seconds spoken at a natural pace)."
))


# ── Image encoding helper ─────────────────────────────────────────────────────

_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # keep in sync with backend/routers/chat.py's cap


def _to_data_uri(image_path: str) -> str:
    """
    Return a base64 data URI for *image_path*, or pass through if already a URI.

    Validates extension/existence/size directly here (defense-in-depth): the
    FastAPI backend already validates uploads before ever reaching the agent,
    but __main__.py's CLI passes a raw local path with no such checks, and a
    node shouldn't implicitly trust that every caller sanitized image_path.
    """
    if image_path.startswith("data:"):
        return image_path
    path = Path(image_path)
    if path.suffix.lower() not in _ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    size = path.stat().st_size
    if size > _MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large: {size} bytes (max {_MAX_IMAGE_BYTES})")
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
    }
    mime = mime_map[path.suffix.lower()]
    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode()
    return f"data:{mime};base64,{encoded}"


_BINOMIAL_RE = re.compile(r"\(([A-Z][a-z]+)\s+([a-z-]+)")


def parse_binomial(species: str) -> tuple[str, str]:
    """
    Extract (genus, species_epithet) from a "Common Name (Genus species)" string.

    Derived deterministically from Gemini's existing `species` output rather than
    asking for genus/epithet as separate structured-output fields — avoids a second
    LLM-populated field that could drift from (or fail validation independently of)
    the scientific name already embedded in `species`.

    Returns ("", "") if no parenthesised binomial is present (e.g. "unknown" on
    an analysis error/low-confidence stub).
    """
    match = _BINOMIAL_RE.search(species or "")
    return (match.group(1), match.group(2)) if match else ("", "")


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Analyse image
# ══════════════════════════════════════════════════════════════════════════════

def node_analyze_image(
    state: SafariGuideState,
    llm: BaseChatModel,
) -> dict:
    """
    Multimodal Gemini call → structured WildlifeIdentification.

    current_analysis is ALWAYS set to this turn's raw result (success or
    error stub) — route_after_analysis and node_unclear_photo_fallback read
    this, never identification_result, so a blurry/failed follow-up photo
    can never clobber the last confidently-identified animal.

    identification_history accumulates (via operator.add) on every successful
    analysis regardless of confidence — unchanged from prior behavior.
    identification_result (last-known-good, read by retrieval/persona) is
    only updated when confidence_score >= MIN_CONFIDENCE.

    Gemini's live threat_level is escalated (never downgraded) against
    species_list.json's curated ground truth here — before identification_result
    AND identification_history are built — so both stay consistent (see
    species_lookup.py). threat_level is exposed to callers (e.g. the API
    response) for their own use; the agent no longer narrates a safety
    warning itself.

    genus/species_epithet are derived deterministically from Gemini's
    `species` string via parse_binomial() rather than requested as separate
    structured-output fields.
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
        ident["genus"], ident["species_epithet"] = parse_binomial(ident["species"])
        curated_threat = ground_truth_threat_level(ident["species"])
        if curated_threat and _THREAT_RANK.get(curated_threat, 0) > _THREAT_RANK.get(ident["threat_level"], 0):
            log.warning(
                "   → Curated ground truth (%s) escalates Gemini's live call (%s) for %r",
                curated_threat, ident["threat_level"], ident["species"],
            )
            ident["threat_level"] = curated_threat

        out = {
            "current_analysis":       ident,
            "identification_history": [ident],   # appended via operator.add
            "error_message":          "",
        }
        if ident["confidence_score"] >= MIN_CONFIDENCE:
            out["identification_result"] = ident
            out["chat_history"] = [
                HumanMessage(content=f"[Photo submitted: {state['image_path']}]"),
                AIMessage(
                    content=(
                        f"Identified **{result.species}** — "
                        f"{result.confidence_score:.0%} confidence, {ident['threat_level']} threat."
                    )
                ),
            ]
        return out

    except Exception as exc:
        log.error("   → analyze_image failed: %s", exc)
        return {
            "current_analysis": {"confidence_score": 0.0, "species": "unknown"},
            "error_message":    str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Unclear photo fallback
# ══════════════════════════════════════════════════════════════════════════════

def node_unclear_photo_fallback(state: SafariGuideState) -> dict:
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
# NODE 3 — RAG retrieval
# ══════════════════════════════════════════════════════════════════════════════

def node_retrieve_information(
    state: SafariGuideState,
    retriever: BaseRetriever,
) -> dict:
    """
    Hybrid BM25 + semantic + web retrieval for verified guidebook facts.
    Query blends species name + user_message for context-aware retrieval
    on text follow-up turns. On a fresh photo identification (no follow_up
    yet), the query is biased toward diet/circadian-rhythm topics instead of
    just the species name, since that's the biographical info the persona
    node is expected to lead with and the curated corpus has no dedicated
    field for it — see node_generate_guide_persona.
    """
    log.info("▶ NODE  retrieve_information")
    species     = state.get("identification_result", {}).get("species", "")
    # Canonicalize against species_list.json first so casing/whitespace drift in
    # Gemini's freeform output doesn't silently drop the species filter (falls
    # back to the naive split for genuinely unlisted/novel species — see #10).
    common_name = (canonical_common_name(species) if species else None) or species.split("(")[0].strip()
    follow_up   = state.get("user_message", "")
    if follow_up:
        query = f"{common_name} {follow_up}".strip()
    else:
        query = f"{common_name} diet feeding behavior circadian rhythm daily activity pattern".strip()
    query = query or "safari wildlife"

    if isinstance(retriever, _EnsembleRetriever):
        docs = retriever.retrieve(query, species=common_name or None)
    else:
        docs = retriever.invoke(query)
    facts = "\n\n---\n\n".join(_format_fact(d) for d in docs)
    log.info("   → %d docs retrieved | query: '%s'", len(docs), query)
    return {"retrieved_facts": facts}


def _format_fact(doc) -> str:
    """
    Label each fact by provenance so the persona LLM can tell curated
    guidebook data (vetted at ingest time) apart from live Tavily web
    results, and prefer the former on conflict — see node_generate_guide_persona.
    """
    if doc.metadata.get("source") == "web":
        label = doc.metadata.get("title") or doc.metadata.get("url") or "unknown page"
        return f"[Source: Web — {label}]\n{doc.page_content}"
    species = doc.metadata.get("species") or "Guidebook"
    return f"[Source: Guidebook — {species}]\n{doc.page_content}"


# ── Shared history-marker helpers (used by NODE 5 and NODE 6 below) ───────────

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


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Summarise history  (long-range memory management)
# ══════════════════════════════════════════════════════════════════════════════

def node_summarize_history(
    state: SafariGuideState,
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

    response = llm.invoke([prompt])
    log.info("   → Conversation summary updated (%d words)", len(response.content.split()))
    return {"conversation_summary": response.content, "summarized_upto": boundary}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Generate guide persona script
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
        task = HumanMessage(content=(
            f"The tourist is asking: \"{follow_up}\"\n\n"
            f"Relevant facts (Guidebook = vetted internal data; Web = live search, "
            f"supplementary only — prefer Guidebook on conflict, especially for "
            f"safety/danger information):\n{facts}{animals_digest}\n\n"
            "Answer as Baako. If the question refers to a previous animal, "
            "use the session memory and animals list above."
        ))
    else:
        trait_line  = ", ".join(traits) if traits else "its distinctive features"
        binomial_line = (
            f"Genus: {genus}. Species: {species_epithet}.\n"
            if genus and species_epithet else ""
        )
        task = HumanMessage(content=(
            f"You have just spotted a {species}! "
            f"{binomial_line}"
            f"Observable traits: {trait_line}.\n\n"
            f"Verified facts (Guidebook = vetted internal data; Web = live search, "
            f"supplementary only — prefer Guidebook on conflict):\n{facts}{animals_digest}\n\n"
            "Generate an audio tour-guide script as Baako introducing this animal. "
            "Clearly state its common name, genus, and species. Then highlight its "
            "circadian rhythm (when it's active) and its diet, drawing only from the "
            "facts above. If the facts above don't cover its circadian rhythm or diet, "
            "say so briefly and respectfully — apologize that this specific detail "
            "isn't available yet rather than guessing or inventing it."
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
# NODE 6 — Generate audio  (conditional on voice_requested)
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
