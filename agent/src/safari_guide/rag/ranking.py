"""
Fusion and ranking: combines multiple sub-retrievers' ranked lists via
Reciprocal Rank Fusion (RRF), with an optional cross-encoder re-rank pass.

This is the algorithm layer — it doesn't know how to talk to Pinecone or
Tavily, only how to merge and score `Document` lists that `backends.py`'s
retrievers hand it.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .backends import _TavilyRetriever

log = logging.getLogger(__name__)


def _doc_key(doc: Document) -> tuple[str, str | None] | tuple[str | None, str | None, str | None] | int:
    """
    Stable cross-retriever identity for a document, used to fuse duplicate
    hits from BM25, Pinecone, and web search into a single scored entry.

    Keyed on the same (species, section, source) triple used at ingest time
    (see `_pinecone_vector_id()`), rather than a content prefix — two distinct
    chunks that happen to share their first N characters must not collide.
    Falls back to a content hash only for documents with no identifying
    metadata at all (e.g. the `_MOCK_DOCUMENTS` corpus).

    Web-search results are keyed separately on their URL: every `_TavilyRetriever`
    doc shares the same (species=None, section=None, source="web") triple, which
    would otherwise collapse all web hits from one query into a single fused
    entry and silently drop the rest.
    """
    metadata = doc.metadata or {}
    species = metadata.get("species")
    section = metadata.get("section")
    source  = metadata.get("source")
    if source == "web":
        url = metadata.get("url")
        return ("web", url) if url else hash(doc.page_content)
    if not species and not section and not source:
        return hash(doc.page_content)
    return (species, section, source)


def _bm25_search(bm25: BM25Retriever, query: str, n: int, species: str) -> list[Document]:
    """
    Species-filtered BM25 search.

    `BM25Retriever` has no metadata-filter API, and its result count is baked
    into `self.k` at construction time. That instance is a long-lived
    singleton shared across concurrent requests (see `backend/main.py`'s
    startup `init_rag()` call), so mutating `self.k` per-call to over-retrieve
    would race. `vectorizer.get_top_n` takes `n` as a stateless, explicit
    argument instead — over-retrieve here and post-filter without ever
    touching shared instance state.
    """
    tokens = bm25.preprocess_func(query)
    docs   = bm25.vectorizer.get_top_n(tokens, bm25.docs, n=n)
    return [d for d in docs if d.metadata.get("species") == species]


class _EnsembleRetriever(BaseRetriever):
    """Minimal Reciprocal Rank Fusion retriever combining multiple sub-retrievers."""

    retrievers: list[Any]
    weights: list[float]
    rrf_k: int = 60
    final_k: int = 6
    cross_encoder: Any | None = None
    rerank_threshold: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _fused_retrieve(
        self,
        query: str,
        species: str | None,
        web_cache: dict[str, list[Document]] | None = None,
    ) -> list[Document]:
        scores:  dict[Any, float]    = {}
        doc_map: dict[Any, Document] = {}

        def _run(retriever) -> list[Document]:
            try:
                if species and hasattr(retriever, "similarity_search"):
                    return retriever.similarity_search(query, filter={"species": species})
                elif species and isinstance(retriever, BM25Retriever):
                    return _bm25_search(retriever, query, n=15, species=species)
                elif web_cache is not None and isinstance(retriever, _TavilyRetriever):
                    # Tavily ignores `species` (no filtering), so the two `retrieve()`
                    # fusion passes (species-filtered, then unfiltered-if-empty) would
                    # otherwise issue the identical web query twice — cache it.
                    if query not in web_cache:
                        web_cache[query] = retriever.invoke(query)
                    return web_cache[query]
                else:
                    return retriever.invoke(query)
            except Exception as exc:
                log.warning(f"Retriever {retriever!r} failed for query {query!r}: {exc}")
                return []

        # Retrievers run concurrently (BM25 is in-memory but Pinecone/Tavily are
        # network calls) — futures are submitted and read back in `self.retrievers`
        # order, NOT completion order, so the "first retriever wins the stored
        # Document on collision" tie-break below stays deterministic regardless
        # of which network call happens to return first.
        with ThreadPoolExecutor(max_workers=len(self.retrievers)) as pool:
            futures = [pool.submit(_run, retriever) for retriever in self.retrievers]
            all_docs = [future.result() for future in futures]

        for docs, weight in zip(all_docs, self.weights):
            for rank, doc in enumerate(docs):
                key = _doc_key(doc)
                # First retriever to produce a given key wins the stored Document
                # (BM25 is listed first in init_rag's `retrievers=[...]` and holds
                # full untruncated content, vs. Pinecone's truncated-at-ingest text) —
                # only the score accumulates across sources, not the content.
                if key not in doc_map:
                    doc_map[key] = doc
                    scores[key]  = 0.0
                scores[key] += weight / (self.rrf_k + rank + 1)

        ranked = [doc_map[k] for k in sorted(scores, key=scores.__getitem__, reverse=True)]

        if self.cross_encoder is not None:
            return self._rerank(query, ranked)

        return ranked[: self.final_k]

    def _rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        """
        Re-score RRF-fused candidates with a cross-encoder, which scores the
        (query, document) pair jointly — unlike RRF/BM25/cosine, which score
        each side independently, then compare. RRF scores are always
        positive, so without this an off-domain query still returns
        top-ranked-but-irrelevant chunks; the threshold lets those return [].
        """
        if not candidates:
            return []
        pairs     = [(query, doc.page_content) for doc in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        reranked  = sorted(zip(ce_scores, candidates), key=lambda pair: pair[0], reverse=True)
        kept      = [doc for score, doc in reranked if score >= self.rerank_threshold]
        return kept[: self.final_k]

    def retrieve(self, query: str, species: str | None = None) -> list[Document]:
        """
        Species-scoped retrieval with a soft fallback: if filtering to
        `species` yields nothing (identification/metadata mismatch, or a
        genuinely novel query), re-run once unfiltered rather than returning
        an empty context to the caller.
        """
        web_cache: dict[str, list[Document]] = {}
        docs = self._fused_retrieve(query, species, web_cache=web_cache)
        if species is not None and not docs:
            log.info(f"No docs for species={species!r} — falling back to unfiltered retrieval")
            docs = self._fused_retrieve(query, None, web_cache=web_cache)
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.retrieve(query, species=None)
