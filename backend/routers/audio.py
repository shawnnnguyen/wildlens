from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Form, Header, HTTPException
from fastapi.responses import FileResponse

from wildlens.tts import synthesise_audio

from ..audio_store import AUDIO_DIR, resolve_audio_path, store_audio
from ..dependencies import get_session_registry
from ..schemas import AudioSynthesizeResponse, ErrorDetail, ErrorResponse
from ..session_registry import SessionRegistry

log = logging.getLogger("backend.routers.audio")

router = APIRouter(tags=["audio"])

_MAX_SYNTHESIZE_TEXT_LENGTH = 2000


def _validate_filename(filename: str) -> None:
    """Reject filenames that could escape AUDIO_DIR via path traversal."""
    if any(c in filename for c in ("/", "\\")):
        _bad_filename()
    if ".." in filename.split("."):
        _bad_filename()
    if not filename.endswith(".mp3"):
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="UNSUPPORTED_FORMAT",
                    message="Only .mp3 files are served here.",
                    field="filename",
                )
            ).model_dump(),
        )


def _bad_filename() -> None:
    raise HTTPException(
        status_code=400,
        detail=ErrorResponse(
            error=ErrorDetail(code="INVALID_FILENAME", message="Invalid filename.")
        ).model_dump(),
    )


@router.get("/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    _validate_filename(filename)

    path = resolve_audio_path(filename)

    # Defence-in-depth: confirm the resolved path stays inside AUDIO_DIR
    # (guards against any symlink or OS-level tricks not caught above)
    try:
        path.resolve().relative_to(AUDIO_DIR.resolve())
    except ValueError:
        _bad_filename()

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="AUDIO_NOT_FOUND",
                    message=f"Audio file '{filename}' not found.",
                )
            ).model_dump(),
        )

    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename=filename,
    )


@router.post("/audio/synthesize", response_model=AudioSynthesizeResponse)
async def synthesize_audio(
    thread_id: str = Form(...),
    text: str = Form(...),
    x_session_secret: str | None = Header(default=None, alias="X-Session-Secret"),
    registry: SessionRegistry = Depends(get_session_registry),
) -> AudioSynthesizeResponse:
    """
    On-demand TTS for text the client already has (e.g. a chat reply already
    rendered on screen) — bypasses the LangGraph turn entirely, unlike the
    voice_requested path on POST /chat which the CLI still uses.
    """
    if not registry.verify(thread_id, x_session_secret):
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_SESSION_SECRET",
                    message="Missing or incorrect X-Session-Secret for this thread_id.",
                ),
                thread_id=thread_id,
            ).model_dump(),
        )

    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error=ErrorDetail(code="NO_INPUT", message="text must not be empty.", field="text"),
                thread_id=thread_id,
            ).model_dump(),
        )
    if len(text) > _MAX_SYNTHESIZE_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="TEXT_TOO_LONG",
                    message=f"text must be <= {_MAX_SYNTHESIZE_TEXT_LENGTH} characters.",
                    field="text",
                ),
                thread_id=thread_id,
            ).model_dump(),
        )

    audio_path = await asyncio.to_thread(synthesise_audio, text)
    if not audio_path or audio_path == "NO_TTS_ENGINE_INSTALLED":
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error=ErrorDetail(code="TTS_UNAVAILABLE", message="No TTS engine installed."),
                thread_id=thread_id,
            ).model_dump(),
        )

    try:
        filename = store_audio(audio_path)
    except Exception:
        log.exception("Failed to store synthesized audio file: %s", audio_path)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(code="AUDIO_STORAGE_FAILED", message="Failed to store audio."),
                thread_id=thread_id,
            ).model_dump(),
        )

    return AudioSynthesizeResponse(audio_url=f"/api/audio/{filename}")
