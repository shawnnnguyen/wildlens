"""
Unit tests for the JSON logging formatter (Phase 3 observability hardening).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildlens.logging_config import JsonFormatter, _ThreadIdFilter, configure_logging, current_thread_id


def _make_record(msg: str = "hello", level: int = logging.INFO, exc_info=None) -> logging.LogRecord:
    return logging.LogRecord(
        name="wildlens.test", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=exc_info,
    )


def test_formatter_emits_valid_json_with_expected_keys():
    record = _make_record("hello world")
    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "wildlens.test"
    assert "timestamp" in payload
    assert "thread_id" not in payload  # no filter attached, no contextvar set


def test_formatter_includes_exception_info():
    try:
        raise ValueError("boom")
    except ValueError:
        record = _make_record("failed", level=logging.ERROR, exc_info=sys.exc_info())

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_thread_id_filter_attaches_contextvar_when_set():
    token = current_thread_id.set("thread-abc")
    try:
        record = _make_record()
        _ThreadIdFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
        assert payload["thread_id"] == "thread-abc"
    finally:
        current_thread_id.reset(token)


def test_thread_id_filter_omits_field_when_unset():
    assert current_thread_id.get() is None  # no requests have set it in this test
    record = _make_record()
    _ThreadIdFilter().filter(record)
    payload = json.loads(JsonFormatter().format(record))
    assert "thread_id" not in payload


def test_configure_logging_is_a_noop_under_pytest():
    """
    Called for real (not mocked) — this test IS running under pytest, so it
    exercises the actual "pytest" in sys.modules guard. Without it, the
    first call to configure_logging() from any future test that imports
    backend.main/agent's __main__.py would replace the root logger's
    handlers process-wide, silently breaking caplog-based assertions in
    every test collected afterward.
    """
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    configure_logging()

    assert root.handlers == handlers_before
    assert not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)
