from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class WildlifeIdentificationOut(BaseModel):
    species: str
    genus: str
    species_epithet: str
    confidence_score: float
    visual_traits: list[str]
    threat_level: Literal["low", "medium", "high"]
    habitat_context: str


class ChatResponse(BaseModel):
    thread_id: str
    session_secret: str | None  # only set on the turn that creates thread_id; store and
                                 # resend as the X-Session-Secret header on every later call
    trace_id: str | None  # Langfuse trace ID for this turn; None when tracing is disabled.
                           # Anchors POST /feedback to the exact LLM response it's rating.
    final_script: str
    audio_url: str | None
    identification: WildlifeIdentificationOut | None
    fallback_triggered: bool
    retrieved_facts: list[str]
    error_message: str | None


class AudioSynthesizeResponse(BaseModel):
    audio_url: str


class FeedbackRating(str, Enum):
    up = "up"
    down = "down"


class FeedbackRequest(BaseModel):
    thread_id: str
    trace_id: str
    rating: FeedbackRating
    comment: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool


class MessageRole(str, Enum):
    human = "human"
    ai = "ai"


class ChatMessageOut(BaseModel):
    role: MessageRole
    content: str


class SessionHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ChatMessageOut]
    conversation_summary: str | None
    identification_history: list[WildlifeIdentificationOut]
    total_turns: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    rag_backend: Literal["bm25+pinecone", "bm25_only", "unavailable"]
    tts_backend: Literal["edge-tts", "gtts", "none"]
    graph_ready: bool


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    thread_id: str | None = None
