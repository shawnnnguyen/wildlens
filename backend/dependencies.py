from __future__ import annotations

from fastapi import Request

from .session_registry import SessionRegistry


def get_graph(request: Request):
    return request.app.state.graph


def get_session_registry(request: Request) -> SessionRegistry:
    return request.app.state.session_registry


def get_rag_backend(request: Request) -> str:
    return request.app.state.rag_backend


def get_tts_backend(request: Request) -> str:
    return request.app.state.tts_backend
