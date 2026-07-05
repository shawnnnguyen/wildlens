from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from wild_lens.graphs import make_turn_input
from wild_lens.observability import invoke_with_tracing

from ..audio_store import store_audio
from ..dependencies import get_graph, get_langfuse_handler, get_session_registry
from ..schemas import (
    ChatResponse,
    ErrorDetail,
    ErrorResponse,
    WildlifeIdentificationOut,
)
from ..session_registry import SessionRegistry

log = logging.getLogger("backend.routers.chat")

router = APIRouter(tags=["chat"])

_ACCEPTED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/heic"})
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


def _build_data_uri(content_type: str, data: bytes) -> str:
    return f"data:{content_type};base64,{base64.b64encode(data).decode()}"


def _split_facts(raw: str) -> list[str]:
    if not raw:
        return []
    return [f.strip() for f in raw.split("\n\n---\n\n") if f.strip()]


def _build_identification(
    result: dict,
    image_provided: bool,
    fallback_triggered: bool,
) -> WildlifeIdentificationOut | None:
    if not image_provided or fallback_triggered or not result:
        return None
    try:
        return WildlifeIdentificationOut(
            species=result.get("species", ""),
            confidence_score=result.get("confidence_score", 0.0),
            visual_traits=result.get("visual_traits", []),
            threat_level=result.get("threat_level", "low"),
            habitat_context=result.get("habitat_context", ""),
        )
    except Exception:
        return None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    thread_id: str = Form(...),
    message: str | None = Form(None),
    voice_requested: bool = Form(False),
    image: UploadFile | None = File(None),
    graph=Depends(get_graph),
    registry: SessionRegistry = Depends(get_session_registry),
    langfuse_handler=Depends(get_langfuse_handler),
) -> ChatResponse:
    # ── Input validation ──────────────────────────────────────────────────────
    if not message and image is None:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error=ErrorDetail(code="NO_INPUT", message="Provide a message or an image."),
                thread_id=thread_id,
            ).model_dump(),
        )

    # ── Image processing ──────────────────────────────────────────────────────
    image_path = ""
    if image is not None:
        if image.content_type not in _ACCEPTED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="UNSUPPORTED_FORMAT",
                        message=f"Accepted types: {', '.join(sorted(_ACCEPTED_MIME_TYPES))}",
                        field="image",
                    ),
                    thread_id=thread_id,
                ).model_dump(),
            )

        data = await image.read()

        if len(data) > _MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="IMAGE_TOO_LARGE",
                        message="Image must be ≤ 10 MB.",
                        field="image",
                    ),
                    thread_id=thread_id,
                ).model_dump(),
            )

        # Pass as a data URI — nodes.py/_to_data_uri() passes these straight through
        image_path = _build_data_uri(image.content_type, data)

    # ── Graph invocation ──────────────────────────────────────────────────────
    turn_input = make_turn_input(
        image_path=image_path,
        user_message=message or "",
        voice_requested=voice_requested,
    )
    config = {"configurable": {"thread_id": thread_id}}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        config["metadata"] = {
            "langfuse_session_id": thread_id,
            "langfuse_tags": [
                "photo_turn" if image_path else "text_turn",
                *(["voice_requested"] if voice_requested else []),
            ],
        }

    try:
        result: dict = await asyncio.to_thread(
            invoke_with_tracing, graph, turn_input, config, langfuse_handler
        )
    except Exception:
        log.exception("Graph invocation failed for thread_id=%s", thread_id)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(code="GRAPH_ERROR", message="Graph invocation failed."),
                thread_id=thread_id,
            ).model_dump(),
        )

    # ── Extract state fields ──────────────────────────────────────────────────
    final_script = result.get("final_script", "")
    audio_file_path = result.get("audio_file_path", "")
    raw_facts = result.get("retrieved_facts", "")
    raw_error = result.get("error_message", "") or ""
    identification_result = result.get("identification_result") or {}

    # node_unclear_photo_fallback sets this exact sentinel; other errors are real faults
    fallback_triggered = raw_error == "low_confidence"
    error_message: str | None = None if (not raw_error or fallback_triggered) else raw_error

    # ── Audio: move temp file into serving directory ──────────────────────────
    audio_url: str | None = None
    if voice_requested:
        if not audio_file_path or audio_file_path == "NO_TTS_ENGINE_INSTALLED":
            if not error_message:
                error_message = "TTS unavailable: no engine installed."
        else:
            try:
                filename = store_audio(audio_file_path)
                audio_url = f"/api/audio/{filename}"
            except Exception:
                log.exception("Failed to store audio file: %s", audio_file_path)
                if not error_message:
                    error_message = "Audio storage failed."

    # ── Register session so history/delete endpoints can find it ─────────────
    registry.register(thread_id)

    return ChatResponse(
        thread_id=thread_id,
        final_script=final_script,
        audio_url=audio_url,
        identification=_build_identification(
            identification_result,
            image_provided=bool(image_path),
            fallback_triggered=fallback_triggered,
        ),
        fallback_triggered=fallback_triggered,
        retrieved_facts=_split_facts(raw_facts),
        error_message=error_message,
    )
