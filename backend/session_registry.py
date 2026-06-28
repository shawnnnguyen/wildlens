from __future__ import annotations

import threading


class SessionRegistry:
    """
    Tracks which thread_ids the API considers active.

    MemorySaver has no public eviction API, so this registry is the
    authoritative source of truth for session existence from the API's
    perspective. Evicted thread_ids remain in the checkpointer's internal
    store but are invisible to all route handlers.

    All operations use a threading.Lock (not asyncio.Lock) so they are safe
    to call from both sync and async contexts without blocking the event loop
    — set/dict operations complete in O(1) and hold the lock for nanoseconds.
    """

    def __init__(self) -> None:
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def register(self, thread_id: str) -> None:
        with self._lock:
            self._active.add(thread_id)

    def evict(self, thread_id: str) -> None:
        """Raise KeyError if thread_id is not registered."""
        with self._lock:
            if thread_id not in self._active:
                raise KeyError(thread_id)
            self._active.discard(thread_id)

    def exists(self, thread_id: str) -> bool:
        with self._lock:
            return thread_id in self._active
