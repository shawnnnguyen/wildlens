"""
Integration tests for the per-session capability token (Phase 1 hardening).

Builds a minimal FastAPI app wired directly to chat.router/sessions.router
with dependency_overrides — no lifespan, no real LLM/RAG/Langfuse calls —
so these run without any API keys or network access.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))                          # for `backend`
sys.path.insert(0, str(_REPO_ROOT / "agent" / "src"))         # for `wildlens`

from backend.dependencies import get_graph, get_langfuse_handler, get_session_registry
from backend.routers import chat, sessions
from backend.session_registry import SessionRegistry


def _fake_graph() -> MagicMock:
    graph = MagicMock()
    graph.invoke.return_value = {
        "final_script": "Hello, tourist!",
        "audio_file_path": "",
        "retrieved_facts": "",
        "error_message": "",
        "identification_result": {},
    }
    graph.get_state.return_value = MagicMock(
        values={"chat_history": [], "conversation_summary": "", "identification_history": []}
    )
    return graph


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    app.include_router(sessions.router, prefix="/api")

    # Mirrors backend/main.py's handler: routers raise HTTPException with an
    # ErrorResponse dict as detail, unwrapped directly (not nested under "detail").
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    registry = SessionRegistry(str(tmp_path / "sessions.db"))
    graph = _fake_graph()
    app.dependency_overrides[get_graph] = lambda: graph
    app.dependency_overrides[get_session_registry] = lambda: registry
    app.dependency_overrides[get_langfuse_handler] = lambda: None

    with TestClient(app) as c:
        c.graph = graph  # stash for assertions
        yield c


def _chat(client, thread_id, message="hi", secret=None):
    headers = {"X-Session-Secret": secret} if secret else {}
    return client.post("/api/chat", data={"thread_id": thread_id, "message": message}, headers=headers)


def test_first_chat_call_issues_a_session_secret(client):
    resp = _chat(client, "sess_a")
    assert resp.status_code == 200
    secret = resp.json()["session_secret"]
    assert secret  # non-empty


def test_failed_first_call_does_not_permanently_lock_out_the_thread_id(client):
    """A brand-new thread_id whose first request fails validation (no message,
    no image) must not leave an orphaned session behind — registry.create()
    already consumed the atomic "first request" slot before validation ran,
    so a retry with the same thread_id must still be treated as a fresh first
    call, not rejected with 403 for a secret that was never handed out."""
    failed = client.post("/api/chat", data={"thread_id": "sess_retry"})  # no message, no image
    assert failed.status_code == 422

    retry = _chat(client, "sess_retry")  # same thread_id, now with a valid message
    assert retry.status_code == 200
    assert retry.json()["session_secret"]  # got a real secret this time


def test_continuing_without_secret_is_rejected(client):
    _chat(client, "sess_b")  # creates the session
    resp = _chat(client, "sess_b")  # no secret this time
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "INVALID_SESSION_SECRET"


def test_continuing_with_wrong_secret_is_rejected(client):
    _chat(client, "sess_c")
    resp = _chat(client, "sess_c", secret="not-the-real-secret")
    assert resp.status_code == 403


def test_continuing_with_correct_secret_succeeds_and_does_not_reissue(client):
    first = _chat(client, "sess_d")
    secret = first.json()["session_secret"]

    second = _chat(client, "sess_d", secret=secret)
    assert second.status_code == 200
    assert second.json()["session_secret"] is None  # only issued once, on creation


def test_two_thread_ids_cannot_use_each_others_secret(client):
    first = _chat(client, "sess_e")
    secret_e = first.json()["session_secret"]

    resp = _chat(client, "sess_f_unrelated", secret=secret_e)
    # sess_f is brand new, so it's created fresh regardless of the (irrelevant)
    # header — the real cross-session check is exercised by the next line.
    assert resp.status_code == 200

    # Now that sess_f exists with its own secret, sess_e's secret must not work on it.
    resp2 = client.post(
        "/api/chat",
        data={"thread_id": "sess_f_unrelated", "message": "again"},
        headers={"X-Session-Secret": secret_e},
    )
    assert resp2.status_code == 403


def test_history_requires_correct_secret(client):
    first = _chat(client, "sess_g")
    secret = first.json()["session_secret"]

    assert client.get("/api/sessions/sess_g/history").status_code == 403
    assert client.get(
        "/api/sessions/sess_g/history", headers={"X-Session-Secret": "wrong"}
    ).status_code == 403
    ok = client.get("/api/sessions/sess_g/history", headers={"X-Session-Secret": secret})
    assert ok.status_code == 200


def test_delete_requires_correct_secret_and_clears_checkpointer(client):
    first = _chat(client, "sess_h")
    secret = first.json()["session_secret"]

    denied = client.delete("/api/sessions/sess_h")
    assert denied.status_code == 403

    ok = client.delete("/api/sessions/sess_h", headers={"X-Session-Secret": secret})
    assert ok.status_code == 204
    client.graph.checkpointer.delete_thread.assert_called_once_with("sess_h")

    # Session is gone — even the correct secret no longer works.
    again = client.delete("/api/sessions/sess_h", headers={"X-Session-Secret": secret})
    assert again.status_code == 404
