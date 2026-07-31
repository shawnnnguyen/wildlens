"""
Unit + integration tests for the per-IP rate limiter (backend/api/rate_limit.py).

Mirrors the pattern in test_session_auth.py: a minimal FastAPI app wired
directly to chat.router, no lifespan, no real LLM/RAG/Langfuse calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend.api.dependencies import get_graph, get_langfuse_handler, get_session_registry
from backend.api.rate_limit import RateLimiter
from backend.api.routers import chat
from backend.api.session_registry import SessionRegistry


def _fake_graph() -> MagicMock:
    graph = MagicMock()
    graph.invoke.return_value = {
        "final_script": "Hello, tourist!",
        "audio_file_path": "",
        "retrieved_facts": "",
        "error_message": "",
        "identification_result": {},
    }
    return graph


def _make_app(tmp_path, chat_rate_limiter: RateLimiter | None) -> FastAPI:
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    registry = SessionRegistry(str(tmp_path / "sessions.db"))
    app.dependency_overrides[get_graph] = lambda: _fake_graph()
    app.dependency_overrides[get_session_registry] = lambda: registry
    app.dependency_overrides[get_langfuse_handler] = lambda: None
    if chat_rate_limiter is not None:
        app.state.chat_rate_limiter = chat_rate_limiter
    return app


def _chat(client: TestClient, thread_id: str):
    return client.post("/api/chat", data={"thread_id": thread_id, "message": "hi"})


def test_requests_within_the_cap_all_succeed(tmp_path):
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    with TestClient(_make_app(tmp_path, limiter)) as client:
        for i in range(3):
            resp = _chat(client, f"sess_{i}")
            assert resp.status_code == 200


def test_request_over_the_cap_is_rejected_with_429(tmp_path):
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    with TestClient(_make_app(tmp_path, limiter)) as client:
        assert _chat(client, "sess_a").status_code == 200
        assert _chat(client, "sess_b").status_code == 200

        third = _chat(client, "sess_c")
        assert third.status_code == 429
        assert third.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"


def test_no_limiter_configured_means_no_limiting(tmp_path):
    # Mirrors test_session_auth.py's app fixture, which never sets
    # app.state.chat_rate_limiter — the dependency must no-op, not error.
    with TestClient(_make_app(tmp_path, chat_rate_limiter=None)) as client:
        for i in range(5):
            resp = _chat(client, f"sess_{i}")
            assert resp.status_code == 200


def test_check_evicts_expired_hits_outside_the_window():
    limiter = RateLimiter(max_requests=1, window_seconds=0.03)
    limiter.check("k")
    with pytest.raises(HTTPException):
        limiter.check("k")

    import time

    time.sleep(0.2)  # comfortably past the window despite Windows timer granularity
    limiter.check("k")  # window elapsed — does not raise
