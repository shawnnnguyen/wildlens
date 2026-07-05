"""
Langfuse observability wiring — shared by the CLI demo and the FastAPI backend.

langfuse>=4 restructured its SDK around a global client (`Langfuse(...)`)
plus a `CallbackHandler` that reads credentials off that already-initialised
client rather than accepting them itself (`CallbackHandler.__init__` only
takes `public_key` / `trace_context` — no `secret_key`, no `host`).
`init_langfuse()` must run exactly once per process, before any
`CallbackHandler()` is constructed.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langfuse.langchain import CallbackHandler

log = logging.getLogger("wild_lens.observability")

_IMAGE_PREFIX = "data:image"
_MAX_STRING_LEN = 2000


def _mask(*, data: Any, **_kwargs: Any) -> Any:
    """
    Redact base64 image payloads and clip oversized strings before Langfuse
    serialises trace input/output/metadata. `data` may be a string, dict,
    list, or nested combination (e.g. serialised chat messages) depending on
    what is being traced, so this walks the structure rather than assuming
    a shape — tourist photos must never reach Langfuse in full.
    """
    if isinstance(data, str):
        if data.startswith(_IMAGE_PREFIX):
            return f"[image omitted, {len(data)} chars]"
        if len(data) > _MAX_STRING_LEN:
            return data[:_MAX_STRING_LEN] + f"...[truncated, {len(data)} chars total]"
        return data
    if isinstance(data, dict):
        return {k: _mask(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask(data=v) for v in data]
    return data


def init_langfuse() -> Optional["CallbackHandler"]:
    """
    Initialise the global Langfuse client and return a LangChain
    CallbackHandler, or None if LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are
    unset. Tracing is optional and must degrade silently, matching how
    Pinecone/Supabase already degrade in rag/factory.py.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            environment=os.getenv("LANGFUSE_ENVIRONMENT", "development"),
            mask=_mask,
        )
        return CallbackHandler()
    except Exception:
        log.exception("Langfuse init failed; continuing without tracing")
        return None


def invoke_with_tracing(graph, turn_input: dict, config: dict, langfuse_handler) -> dict:
    """
    Run one graph turn, wrapped in a parent Langfuse span when tracing is
    enabled so per-turn outcome (species/confidence/threat/error) can be
    attached once the graph finishes — LangGraph's node-level LLM spans
    (produced via langfuse_handler in config["callbacks"]) nest underneath
    automatically. No-ops straight through to graph.invoke() otherwise.
    """
    if langfuse_handler is None:
        return graph.invoke(turn_input, config)

    from langfuse import get_client

    client = get_client()
    with client.start_as_current_observation(name="chat_turn", as_type="span"):
        result = graph.invoke(turn_input, config)
        ident = result.get("identification_result") or {}
        client.update_current_span(
            output={"error_message": result.get("error_message") or None},
            metadata={
                "species": ident.get("species"),
                "confidence_score": ident.get("confidence_score"),
                "threat_level": ident.get("threat_level"),
            },
        )
        return result
