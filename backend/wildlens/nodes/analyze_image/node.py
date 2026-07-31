"""
NODE — Analyse image

Multimodal Gemini vision call → structured WildlifeIdentification.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from wildlens.data.species_lookup import ground_truth_threat_level
from wildlens.nodes._shared import _RETRY_POLICY, _is_truncated
from wildlens.state import MIN_CONFIDENCE, WildlensState, WildlifeIdentification

log = logging.getLogger("safari_guide.nodes")

# Ordering used to escalate (never downgrade) Gemini's live threat_level call
# against species_list.json's curated ground truth — see node_analyze_image.
_THREAT_RANK = {"low": 0, "medium": 1, "high": 2}

_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # keep in sync with backend/api/routers/chat.py's cap


@_RETRY_POLICY
def _invoke_structured_with_retry(structured, prompt: HumanMessage) -> dict:
    """
    Same retry policy as _shared._invoke_with_retry, but wrapping the
    include_raw=True call directly so a malformed/invalid structured response
    (raw_result["parsing_error"] set) gets retried too, not just a network-
    level exception from .invoke() itself. Truncation is NOT retried here —
    it's returned as-is for the caller to detect via _is_truncated, since
    retrying an already-truncated response at the same max_output_tokens is
    unlikely to help and would just add latency for a near-certain repeat.
    """
    raw_result: dict = structured.invoke([prompt])
    if raw_result["parsing_error"] is not None and not _is_truncated(raw_result["raw"]):
        raise raw_result["parsing_error"]
    return raw_result


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


def node_analyze_image(
    state: WildlensState,
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
        structured = llm.with_structured_output(WildlifeIdentification, include_raw=True)
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

        raw_result: dict = _invoke_structured_with_retry(structured, prompt)
        raw_message = raw_result["raw"]

        if _is_truncated(raw_message):
            log.error(
                "   → analyze_image response truncated by token limit (finish_reason=%s)",
                raw_message.response_metadata.get("finish_reason"),
            )
            return {
                "current_analysis": {"confidence_score": 0.0, "species": "unknown"},
                "error_message":    "truncated_response",
            }

        if raw_result["parsing_error"] is not None:
            raise raw_result["parsing_error"]

        result: WildlifeIdentification = raw_result["parsed"]
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
                HumanMessage(content="[Photo submitted]"),
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
