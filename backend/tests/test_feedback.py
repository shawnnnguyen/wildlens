"""
Integration tests for POST /api/feedback's session-secret gate and the
Langfuse-disabled no-op path — mirrors the pattern in
test_audio_synthesize.py, scoped to the feedback router only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))                          # for `backend`
sys.path.insert(0, str(_REPO_ROOT / "agent" / "src"))         # for `wildlens`

from backend.dependencies import get_langfuse_handler, get_session_registry
from backend.routers import feedback
from backend.session_registry import SessionRegistry


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.include_router(feedback.router, prefix="/api")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    registry = SessionRegistry(str(tmp_path / "sessions.db"))
    app.dependency_overrides[get_session_registry] = lambda: registry
    # Non-None sentinel: only its truthiness is checked in the router, real
    # scoring goes through the module-level langfuse.get_client() import,
    # which each test below patches individually.
    app.dependency_overrides[get_langfuse_handler] = lambda: MagicMock()

    with TestClient(app) as c:
        c.registry = registry  # stash for assertions
        yield c


def test_feedback_requires_correct_secret(client):
    client.registry.create("sess_a")

    missing = client.post(
        "/api/feedback", json={"thread_id": "sess_a", "trace_id": "t1", "rating": "up"}
    )
    assert missing.status_code == 403
    assert missing.json()["error"]["code"] == "INVALID_SESSION_SECRET"

    wrong = client.post(
        "/api/feedback",
        json={"thread_id": "sess_a", "trace_id": "t1", "rating": "up"},
        headers={"X-Session-Secret": "not-the-real-secret"},
    )
    assert wrong.status_code == 403


def test_feedback_rejects_unknown_thread_id(client):
    resp = client.post(
        "/api/feedback",
        json={"thread_id": "never-created", "trace_id": "t1", "rating": "down"},
        headers={"X-Session-Secret": "anything"},
    )
    assert resp.status_code == 403


def test_feedback_submits_score_with_deterministic_id(client):
    secret = client.registry.create("sess_b")

    fake_client = MagicMock()
    with patch("langfuse.get_client", return_value=fake_client):
        resp = client.post(
            "/api/feedback",
            json={"thread_id": "sess_b", "trace_id": "trace-123", "rating": "up", "comment": "Great answer!"},
            headers={"X-Session-Secret": secret},
        )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    fake_client.create_score.assert_called_once()
    _, kwargs = fake_client.create_score.call_args
    assert kwargs["trace_id"] == "trace-123"
    assert kwargs["value"] == 1.0
    assert kwargs["comment"] == "Great answer!"
    assert kwargs["score_id"] == feedback._score_id("trace-123")

    # Resubmitting for the same trace_id must reuse the same score_id (upsert,
    # not a duplicate row) regardless of rating/comment changes.
    with patch("langfuse.get_client", return_value=fake_client):
        client.post(
            "/api/feedback",
            json={"thread_id": "sess_b", "trace_id": "trace-123", "rating": "down"},
            headers={"X-Session-Secret": secret},
        )
    second_kwargs = fake_client.create_score.call_args.kwargs
    assert second_kwargs["score_id"] == kwargs["score_id"]
    assert second_kwargs["value"] == 0.0


def test_feedback_noops_when_langfuse_disabled(client):
    secret = client.registry.create("sess_c")
    client.app.dependency_overrides[get_langfuse_handler] = lambda: None

    resp = client.post(
        "/api/feedback",
        json={"thread_id": "sess_c", "trace_id": "t1", "rating": "up"},
        headers={"X-Session-Secret": secret},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
