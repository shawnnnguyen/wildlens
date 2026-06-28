from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..audio_store import AUDIO_DIR, resolve_audio_path
from ..schemas import ErrorDetail, ErrorResponse

router = APIRouter(tags=["audio"])


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
