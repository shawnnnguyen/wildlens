"""
Structured (JSON) logging for the plain-Python code Langfuse doesn't trace.

Langfuse already captures LLM-call spans, token usage/cost, and per-turn
species/confidence/threat_level metadata (see observability.py) — this module
is for everything else: retry exhaustion, RAG sub-retriever failures, TTS
fallback, startup/shutdown lifecycle. Both the FastAPI backend and the CLI
entrypoint previously ran on unconfigured/plain-text logging (see
configure_logging's docstring); this replaces both with one JSON formatter.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Set by the FastAPI chat router around each request so every log line emitted
# while handling that request — including from agent code that has no idea
# what a "request" is — carries the same thread_id Langfuse already uses as
# langfuse_session_id (see backend/routers/chat.py), making logs and traces
# joinable on one field. asyncio.to_thread() copies contextvars into the
# worker thread it spawns, so this reaches nodes.py/rag code without any
# further plumbing.
current_thread_id: ContextVar[str | None] = ContextVar("current_thread_id", default=None)

_RESERVED_LOG_RECORD_ATTRS = frozenset(logging.LogRecord(
    "", 0, "", 0, "", (), None,
).__dict__.keys()) | {"message", "asctime", "taskName"}


class _ThreadIdFilter(logging.Filter):
    """Attaches the current request's thread_id to every LogRecord, if set."""

    def filter(self, record: logging.LogRecord) -> bool:
        thread_id = current_thread_id.get()
        if thread_id is not None:
            record.thread_id = thread_id
        return True


class JsonFormatter(logging.Formatter):
    """Minimal stdlib-only JSON line formatter — no new dependency.

    Emits timestamp/level/logger/message always; thread_id when the
    _ThreadIdFilter attached one; exc_info when present; and any other
    caller-supplied `extra=` keys (skipping LogRecord's own built-in
    attributes so they aren't duplicated).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        thread_id = getattr(record, "thread_id", None)
        if thread_id is not None:
            payload["thread_id"] = thread_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_ATTRS and key != "thread_id" and key not in payload:
                try:
                    json.dumps(value)
                except TypeError:
                    value = repr(value)
                payload[key] = value
        return json.dumps(payload, default=str)


_configured = False


def configure_logging(level: str | None = None) -> None:
    """
    Install the JSON formatter on the root logger.

    Idempotent — safe to call more than once (e.g. backend.main gets
    imported both by uvicorn and by pytest's test client), so a second call
    is a no-op rather than fighting pytest's caplog handler with a
    force=True reconfigure.

    Also a no-op under pytest (detected via `"pytest" in sys.modules`, true
    for the whole test-process lifetime, not just mid-test — unlike
    PYTEST_CURRENT_TEST). No test currently imports backend.main/__main__.py
    at collection time, so this guard is a no-op today, but the FIRST call
    (not just repeats) needs to be skipped under pytest: it would otherwise
    replace the root logger's handlers process-wide for the rest of the
    session the moment such an import happened, silently breaking any
    caplog-based assertion in every test file collected afterward.
    """
    global _configured
    if _configured:
        return
    _configured = True

    if "pytest" in sys.modules:
        return

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_ThreadIdFilter())

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.handlers = [handler]


def uvicorn_log_config(level: str | None = None) -> dict:
    """
    A uvicorn `log_config` dict that routes uvicorn's own loggers (which
    install handlers with propagate=False, so they're invisible to a root
    logger handler swap) through the same JsonFormatter — otherwise backend
    stdout is a mix of JSON app logs and uvicorn's plain-text access/error
    lines.
    """
    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": "wildlens.logging_config.JsonFormatter"},
        },
        "handlers": {
            "default": {"class": "logging.StreamHandler", "formatter": "json"},
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": resolved_level, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": resolved_level, "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": resolved_level, "propagate": False},
        },
    }
