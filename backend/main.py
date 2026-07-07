from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from wildlens.graphs import build_graph
from wildlens.observability import init_langfuse
from wildlens.rag import _NullRetriever, init_rag
from wildlens.tts import _EDGE_TTS_AVAILABLE, _GTTS_AVAILABLE

from .audio_store import audio_janitor, ensure_audio_dir
from .routers import audio, chat, health, sessions
from .schemas import ErrorDetail, ErrorResponse
from .session_registry import SessionRegistry

load_dotenv()

log = logging.getLogger("backend")

_REQUIRED_ENV_VARS = ["GOOGLE_API_KEY", "DEEPSEEK_API_KEY"]
_RECOMMENDED_ENV_VARS = ["PINECONE_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]


def _detect_tts_backend() -> str:
    if _EDGE_TTS_AVAILABLE:
        return "edge-tts"
    if _GTTS_AVAILABLE:
        return "gtts"
    return "none"


def _detect_rag_backend(retriever) -> str:
    """Inspect the ensemble's secondary retriever to determine which backends are live."""
    if not hasattr(retriever, "retrievers") or len(retriever.retrievers) < 2:
        return "unavailable"
    return "bm25_only" if isinstance(retriever.retrievers[1], _NullRetriever) else "bm25+pinecone"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 1. Validate required env vars
    missing = [v for v in _REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    for var in _RECOMMENDED_ENV_VARS:
        if not os.getenv(var):
            log.warning("%s not set — RAG will degrade gracefully", var)

    # 2-3. Instantiate LLMs (imports deferred so missing packages give a clear error)
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI

    llm_vision = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash"), temperature=0.35,
    )
    llm_text = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0.35,
    )

    # 4. Init RAG (slow: loads HuggingFace model + Pinecone; run off the event loop)
    retriever = await asyncio.to_thread(init_rag)
    rag_backend = _detect_rag_backend(retriever)
    log.info("RAG backend: %s", rag_backend)

    # 4b. Init Langfuse (optional; None when LANGFUSE_PUBLIC_KEY/SECRET_KEY unset)
    langfuse_handler = init_langfuse()
    log.info("Langfuse tracing: %s", "enabled" if langfuse_handler else "disabled")

    # 4c. SQLite checkpointer — survives restarts (MemorySaver was pure in-process
    # RAM) and gives DELETE /api/sessions/{id} a real delete_thread() to call.
    # check_same_thread=False: graph.invoke() runs via asyncio.to_thread, so calls
    # land on whichever worker thread the event loop picks, not always the same one.
    from langgraph.checkpoint.sqlite import SqliteSaver

    sessions_db_path = os.getenv("SESSIONS_DB_PATH", "safari_sessions.db")
    sqlite_conn = sqlite3.connect(sessions_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(sqlite_conn)
    await asyncio.to_thread(checkpointer.setup)

    # 5. Build compiled graph
    graph = await asyncio.to_thread(
        build_graph,
        llm_vision, llm_text, retriever,
        tracing_enabled=bool(langfuse_handler),
        checkpointer=checkpointer,
    )

    # 6. Detect TTS (import-time flags; no I/O)
    tts_backend = _detect_tts_backend()
    log.info("TTS backend: %s", tts_backend)

    # 7-8. Store singletons on app.state
    app.state.graph = graph
    app.state.rag_backend = rag_backend
    app.state.tts_backend = tts_backend
    app.state.langfuse_handler = langfuse_handler
    # Same SQLite file as the checkpointer above — session secrets and graph
    # state live side by side and share the same restart-survives guarantee.
    app.state.session_registry = SessionRegistry(sessions_db_path)

    # 9. Ensure audio serving directory exists
    ensure_audio_dir()

    # 10. Start audio janitor
    ttl      = int(os.getenv("AUDIO_TTL_SECONDS", "3600"))
    interval = int(os.getenv("JANITOR_INTERVAL_SECONDS", "900"))
    janitor  = asyncio.create_task(audio_janitor(ttl, interval))

    log.info("Safari Guide backend ready")
    yield

    # Shutdown: cancel the sleeping janitor cleanly
    janitor.cancel()
    try:
        await janitor
    except asyncio.CancelledError:
        pass
    sqlite_conn.close()
    log.info("Safari Guide backend shutting down")


def create_app() -> FastAPI:
    app = FastAPI(title="Safari Guide API", version="1.0.0", lifespan=lifespan)

    # ── Middleware ────────────────────────────────────────────────────────────
    origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
    allowed_origins = ["*"] if origins_raw == "*" else [o.strip() for o in origins_raw.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Session-Secret"],
    )

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        # Routers raise HTTPException with an ErrorResponse dict as detail.
        # Pass it through directly; wrap anything else in the standard envelope.
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(code="HTTP_ERROR", message=str(exc.detail))
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(loc) for loc in first.get("loc", [])) or None
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message=first.get("msg", str(exc)),
                    field=field,
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(code="GRAPH_ERROR", message="An internal error occurred.")
            ).model_dump(),
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api")
    app.include_router(sessions.router, prefix="/api")
    app.include_router(audio.router, prefix="/api")

    return app


app = create_app()
