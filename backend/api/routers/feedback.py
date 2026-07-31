from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, Depends, Header, HTTPException

from ..dependencies import get_langfuse_handler, get_session_registry
from ..schemas import ErrorDetail, ErrorResponse, FeedbackRequest, FeedbackResponse
from ..session_registry import SessionRegistry

log = logging.getLogger("backend.routers.feedback")

router = APIRouter(tags=["feedback"])


def _score_id(trace_id: str) -> str:
    """
    Deterministic score ID for (trace_id, "user_feedback") so a later
    resubmission — the visitor flips their thumb, or adds a note after
    already rating — upserts the same Langfuse score instead of piling up
    duplicate rows for one turn.
    """
    return hashlib.sha256(f"{trace_id}:user_feedback".encode()).hexdigest()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    x_session_secret: str | None = Header(default=None, alias="X-Session-Secret"),
    registry: SessionRegistry = Depends(get_session_registry),
    langfuse_handler=Depends(get_langfuse_handler),
) -> FeedbackResponse:
    """
    Attach human judgment (thumbs up/down + an optional free-text note) to
    the Langfuse trace for one chat turn, as a "user_feedback" Score. Never
    mandatory on the client side; here, it's just an authenticated write —
    same session capability token as every other per-thread endpoint.

    Note: this proves the caller holds thread_id's secret, not that trace_id
    was actually produced by thread_id's own turns (no trace_id<->thread_id
    mapping is persisted server-side to check against). Accepted for now:
    trace_ids are high-entropy and never surfaced to other sessions.
    """
    if not registry.verify(body.thread_id, x_session_secret):
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_SESSION_SECRET",
                    message="Missing or incorrect X-Session-Secret for this thread_id.",
                ),
                thread_id=body.thread_id,
            ).model_dump(),
        )

    if langfuse_handler is None:
        # Tracing disabled process-wide — degrade silently, same convention as
        # observability.init_langfuse()/invoke_with_tracing() and rag/factory.py.
        return FeedbackResponse(ok=True)

    from langfuse import get_client

    try:
        get_client().create_score(
            name="user_feedback",
            value=1.0 if body.rating.value == "up" else 0.0,
            trace_id=body.trace_id,
            data_type="NUMERIC",
            comment=body.comment,
            score_id=_score_id(body.trace_id),
        )
    except Exception:
        log.exception("Failed to submit feedback score for trace_id=%s", body.trace_id)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(code="FEEDBACK_SUBMIT_FAILED", message="Failed to record feedback."),
                thread_id=body.thread_id,
            ).model_dump(),
        )

    return FeedbackResponse(ok=True)
