from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from .schemas import ErrorDetail, ErrorResponse


class RateLimiter:
    """Sliding-window per-key request cap, in-process only.

    Good enough for a single-worker deployment where the goal is putting a
    ceiling on API spend, not exact fairness — swap for a shared store
    (Redis) before running multiple workers/instances, since counters here
    don't survive a restart and aren't shared across processes.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self._window_seconds:
            hits.popleft()
        if len(hits) >= self._max_requests:
            retry_after = int(self._window_seconds - (now - hits[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="RATE_LIMIT_EXCEEDED",
                        message=f"Rate limit exceeded — try again in {retry_after}s.",
                    )
                ).model_dump(),
            )
        hits.append(now)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limit_dependency(state_attr: str):
    """Builds a FastAPI dependency that checks the RateLimiter stored at
    `app.state.<state_attr>`. Apps that never set that attribute (e.g. the
    minimal routers-only apps some tests build directly) skip limiting
    entirely instead of erroring — this only activates when main.py's
    lifespan wires it up.
    """

    async def _dependency(request: Request) -> None:
        limiter: RateLimiter | None = getattr(request.app.state, state_attr, None)
        if limiter is not None:
            limiter.check(_client_key(request))

    return _dependency
