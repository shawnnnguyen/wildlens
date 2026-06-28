from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_graph, get_rag_backend, get_tts_backend
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    graph=Depends(get_graph),
    rag_backend: str = Depends(get_rag_backend),
    tts_backend: str = Depends(get_tts_backend),
) -> HealthResponse:
    graph_ready = graph is not None
    status = "ok" if (graph_ready and rag_backend != "unavailable") else "degraded"
    return HealthResponse(
        status=status,
        rag_backend=rag_backend,
        tts_backend=tts_backend,
        graph_ready=graph_ready,
    )
