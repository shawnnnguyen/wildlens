from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from langchain_core.messages import HumanMessage

from ..dependencies import get_graph, get_session_registry
from ..schemas import (
    ChatMessageOut,
    ErrorDetail,
    ErrorResponse,
    MessageRole,
    SessionHistoryResponse,
    WildlifeIdentificationOut,
)
from ..session_registry import SessionRegistry

log = logging.getLogger("backend.routers.sessions")

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _message_role(msg) -> MessageRole:
    return MessageRole.human if isinstance(msg, HumanMessage) else MessageRole.ai


def _coerce_identification_history(raw: list[dict]) -> list[WildlifeIdentificationOut]:
    out = []
    for item in raw:
        try:
            out.append(WildlifeIdentificationOut(
                species=item.get("species", ""),
                genus=item.get("genus", ""),
                species_epithet=item.get("species_epithet", ""),
                confidence_score=item.get("confidence_score", 0.0),
                visual_traits=item.get("visual_traits", []),
                threat_level=item.get("threat_level", "low"),
                habitat_context=item.get("habitat_context", ""),
            ))
        except Exception:
            continue
    return out


def _check_session(thread_id: str, registry: SessionRegistry) -> None:
    # Runs before _check_session_secret, so an unauthenticated caller can
    # distinguish "no such session" (404) from "wrong secret" (403) — a minor
    # existence oracle. Accepted trade-off: thread_ids are 122-bit random
    # UUIDs (unguessable), and reversing the order would turn "already
    # deleted" into a misleading 403 instead of an honest 404.
    if not registry.exists(thread_id):
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="SESSION_NOT_FOUND",
                    message=f"No active session for thread_id '{thread_id}'.",
                ),
                thread_id=thread_id,
            ).model_dump(),
        )


def _check_session_secret(thread_id: str, secret: str | None, registry: SessionRegistry) -> None:
    if not registry.verify(thread_id, secret):
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


@router.get("/{thread_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    thread_id: str,
    x_session_secret: str | None = Header(default=None, alias="X-Session-Secret"),
    graph=Depends(get_graph),
    registry: SessionRegistry = Depends(get_session_registry),
) -> SessionHistoryResponse:
    _check_session(thread_id, registry)
    _check_session_secret(thread_id, x_session_secret, registry)

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await asyncio.to_thread(graph.get_state, config)
    values: dict = snapshot.values if snapshot else {}

    raw_history = values.get("chat_history", [])
    messages = [
        ChatMessageOut(role=_message_role(msg), content=msg.content)
        for msg in raw_history
        if isinstance(msg.content, str)
    ]

    summary = values.get("conversation_summary") or None

    return SessionHistoryResponse(
        thread_id=thread_id,
        messages=messages,
        conversation_summary=summary,
        identification_history=_coerce_identification_history(
            values.get("identification_history", [])
        ),
        total_turns=len(messages),
    )


@router.delete("/{thread_id}", status_code=204)
async def delete_session(
    thread_id: str,
    x_session_secret: str | None = Header(default=None, alias="X-Session-Secret"),
    graph=Depends(get_graph),
    registry: SessionRegistry = Depends(get_session_registry),
) -> None:
    _check_session(thread_id, registry)
    _check_session_secret(thread_id, x_session_secret, registry)

    try:
        registry.evict(thread_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="SESSION_NOT_FOUND",
                    message=f"No active session for thread_id '{thread_id}'.",
                ),
                thread_id=thread_id,
            ).model_dump(),
        )

    # Actually clear the checkpointer (SqliteSaver/MemorySaver both support
    # delete_thread) — previously only the tracking set was cleared here,
    # leaving the full conversation state behind in the checkpoint store.
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is not None:
        await asyncio.to_thread(checkpointer.delete_thread, thread_id)
