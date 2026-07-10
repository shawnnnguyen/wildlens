"""
Integration tests for POST /api/audio/synthesize's session-secret gate and
input validation — mirrors the pattern in test_session_auth.py, but scoped
to the audio router only (no graph/RAG involved).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))                          # for `backend`
sys.path.insert(0, str(_REPO_ROOT / "agent" / "src"))         # for `wildlens`

from backend.dependencies import get_session_registry
from backend.routers import audio
from backend.session_registry import SessionRegistry


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.include_router(audio.router, prefix="/api")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    registry = SessionRegistry(str(tmp_path / "sessions.db"))
    app.dependency_overrides[get_session_registry] = lambda: registry

    with TestClient(app) as c:
        c.registry = registry  # stash for assertions
        yield c


def test_synthesize_requires_correct_secret(client):
    client.registry.create("sess_a")

    missing = client.post("/api/audio/synthesize", data={"thread_id": "sess_a", "text": "hi"})
    assert missing.status_code == 403
    assert missing.json()["error"]["code"] == "INVALID_SESSION_SECRET"

    wrong = client.post(
        "/api/audio/synthesize",
        data={"thread_id": "sess_a", "text": "hi"},
        headers={"X-Session-Secret": "not-the-real-secret"},
    )
    assert wrong.status_code == 403


def test_synthesize_rejects_unknown_thread_id(client):
    resp = client.post(
        "/api/audio/synthesize",
        data={"thread_id": "never-created", "text": "hi"},
        headers={"X-Session-Secret": "anything"},
    )
    assert resp.status_code == 403


def test_synthesize_succeeds_and_stores_audio(client, tmp_path):
    secret = client.registry.create("sess_b")
    fake_path = str(tmp_path / "fake.mp3")
    Path(fake_path).write_bytes(b"fake-mp3-bytes")

    with patch("backend.routers.audio.synthesise_audio", return_value=fake_path) as mock_synth, \
         patch("backend.routers.audio.store_audio", return_value="fake.mp3") as mock_store:
        resp = client.post(
            "/api/audio/synthesize",
            data={"thread_id": "sess_b", "text": "Hello there."},
            headers={"X-Session-Secret": secret},
        )

    assert resp.status_code == 200
    assert resp.json() == {"audio_url": "/api/audio/fake.mp3"}
    mock_synth.assert_called_once_with("Hello there.")
    mock_store.assert_called_once_with(fake_path)


def test_synthesize_rejects_empty_text(client):
    secret = client.registry.create("sess_c")

    resp = client.post(
        "/api/audio/synthesize",
        data={"thread_id": "sess_c", "text": "   "},
        headers={"X-Session-Secret": secret},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "NO_INPUT"


def test_synthesize_rejects_text_over_length_cap(client):
    secret = client.registry.create("sess_d")

    resp = client.post(
        "/api/audio/synthesize",
        data={"thread_id": "sess_d", "text": "x" * 2001},
        headers={"X-Session-Secret": secret},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "TEXT_TOO_LONG"


def test_synthesize_returns_503_when_tts_unavailable(client):
    secret = client.registry.create("sess_e")

    with patch("backend.routers.audio.synthesise_audio", return_value="NO_TTS_ENGINE_INSTALLED"):
        resp = client.post(
            "/api/audio/synthesize",
            data={"thread_id": "sess_e", "text": "Hello."},
            headers={"X-Session-Secret": secret},
        )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "TTS_UNAVAILABLE"


def test_synthesize_returns_500_when_storage_fails(client, tmp_path):
    secret = client.registry.create("sess_f")
    fake_path = str(tmp_path / "fake2.mp3")
    Path(fake_path).write_bytes(b"fake-mp3-bytes")

    with patch("backend.routers.audio.synthesise_audio", return_value=fake_path), \
         patch("backend.routers.audio.store_audio", side_effect=OSError("disk full")):
        resp = client.post(
            "/api/audio/synthesize",
            data={"thread_id": "sess_f", "text": "Hello."},
            headers={"X-Session-Secret": secret},
        )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "AUDIO_STORAGE_FAILED"
