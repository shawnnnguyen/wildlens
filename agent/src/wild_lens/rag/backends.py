"""
Retriever adapter classes — one per retrieval backend.

_PineconeRetrieverWrapper — semantic search over the Pinecone 'text' namespace.
_NullRetriever           — no-op stand-in used when a backend is unconfigured
                            or fails to initialise, so the ensemble degrades
                            gracefully instead of erroring out.
_TavilyRetriever          — live web search, tagged source="web" so downstream
                            nodes can label facts by provenance.
"""
from __future__ import annotations

import datetime
import logging
import os
import threading
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

log = logging.getLogger(__name__)


class _PineconeRetrieverWrapper(BaseRetriever):
    """
    Holds the raw `PineconeVectorStore` alongside the retained `.invoke()`
    interface, so the unfiltered fusion path in `_EnsembleRetriever` keeps
    working while a species-filtered `similarity_search(..., filter=...)`
    call is also available for later, query-time-filtered retrieval.
    """

    vectorstore: Any
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k or self.k, filter=filter)


class _NullRetriever(BaseRetriever):
    """
    Stand-in for Pinecone retriever when credentials are absent or init fails.
    Returns empty results so _EnsembleRetriever degrades gracefully to BM25-only.
    """

    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return []

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        return []


class _TavilyRetriever(BaseRetriever):
    """
    Live web search via Tavily, wrapped as Documents tagged `source="web"` so
    downstream nodes (see `node_retrieve_information`) can label facts by
    provenance and the persona LLM can be told to prefer vetted guidebook
    facts over these when they conflict.

    Deliberately has no `similarity_search` method and isn't a `BM25Retriever`,
    so it always falls into `_EnsembleRetriever._fused_retrieve`'s plain
    `retriever.invoke(query)` branch — species filtering doesn't apply to a
    live web query, and the species name is already folded into `query` by
    the caller.
    """

    client: Any
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # In-process daily call cap — bounds Tavily spend regardless of how often
    # the gating in _EnsembleRetriever decides the local corpus is thin.
    # Per-process only (not shared across multiple backend workers), same
    # scope limitation as this codebase's other long-lived singletons (e.g.
    # MemorySaver) — acceptable for the current single-instance deployment.
    _call_count: int = PrivateAttr(default=0)
    _count_date: Any = PrivateAttr(default=None)
    _count_lock: Any = PrivateAttr(default_factory=threading.Lock)

    def _under_daily_cap(self) -> bool:
        try:
            cap = int(os.getenv("TAVILY_DAILY_CALL_CAP", "500"))
        except ValueError:
            cap = 500
        today = datetime.date.today()
        with self._count_lock:
            if self._count_date != today:
                self._count_date = today
                self._call_count = 0
            if self._call_count >= cap:
                return False
            self._call_count += 1
            return True

    def _to_documents(self, query: str) -> list[Document]:
        if not self._under_daily_cap():
            log.warning(
                "Tavily daily call cap (%s) reached — skipping web search for %r",
                os.getenv("TAVILY_DAILY_CALL_CAP", "500"), query,
            )
            return []
        results = self.client.search(query, max_results=self.k).get("results", [])
        return [
            Document(
                page_content=r.get("content", ""),
                metadata={"source": "web", "url": r.get("url", ""), "title": r.get("title", "")},
            )
            for r in results
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self._to_documents(query)
