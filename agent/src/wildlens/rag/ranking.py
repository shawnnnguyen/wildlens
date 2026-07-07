"""
Fusion and ranking: combines multiple sub-retrievers' ranked lists via
Reciprocal Rank Fusion (RRF), with an optional cross-encoder re-rank pass.

This is the algorithm layer — it doesn't know how to talk to Pinecone or
Tavily, only how to merge and score `Document` lists that `backends.py`'s
retrievers hand it.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

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

    `preprocess_func`/`vectorizer`/`docs` are undocumented `BM25Retriever`
    internals with no public equivalent for this over-retrieve pattern — if a
    `langchain_community` upgrade renames/removes them, fall back to the
    public `.invoke()` API (capped at `bm25.k`, so over-retrieval for
    species-filtering is lost on this path) rather than crashing retrieval
    entirely. See test_bm25_internal_api_still_present, a canary test that
    fails loudly on such a dependency bump.
    """
    try:
        tokens = bm25.preprocess_func(query)
        docs   = bm25.vectorizer.get_top_n(tokens, bm25.docs, n=n)
    except AttributeError as exc:
        log.warning(
            "BM25Retriever internals changed (%s) — likely a langchain_community "
            "version bump. Falling back to the public .invoke() API.", exc,
        )
        docs = bm25.invoke(query)
    return [d for d in docs if d.metadata.get("species") == species]


class _EnsembleRetriever(BaseRetriever):
    """Minimal Reciprocal Rank Fusion retriever combining multiple sub-retrievers."""

    retrievers: list[Any]
    weights: list[float]
    rrf_k: int = 60
    final_k: int = 6
    cross_encoder: Any | None = None
    rerank_threshold: float = 0.0

    # Enrichment write-back deps (all optional — None disables enrichment
    # entirely, so this retriever still works standalone e.g. in tests or
    # local dev without Supabase/Pinecone). See enrich_async().
    supabase_store: Any | None = None
    pinecone_index: Any | None = None
    embeddings: Any | None = None
    enrichment_rebuild_every: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Lazily created and reused across retrieve() calls — this retriever is a
    # long-lived singleton (see init_rag()'s startup call), so spinning up a
    # new ThreadPoolExecutor per call would waste thread-spawn overhead on
    # every single request. No explicit shutdown(): it lives as long as the
    # process, matching how the Pinecone/Supabase clients elsewhere in this
    # codebase are also never explicitly torn down.
    _executor: ThreadPoolExecutor | None = PrivateAttr(default=None)
    _executor_lock: Any = PrivateAttr(default_factory=threading.Lock)

    # Separate single-worker queue for enrichment writes (see enrich_async) —
    # deliberately not the retrieval `_executor` above, so writes are strictly
    # serialized (no duplicate-write races between concurrent sessions) and
    # never contend with in-flight retrieval fan-out.
    _enrichment_executor: ThreadPoolExecutor | None = PrivateAttr(default=None)
    _enrichment_executor_lock: Any = PrivateAttr(default_factory=threading.Lock)
    _pending_enrichments: int = PrivateAttr(default=0)

    def _get_executor(self) -> ThreadPoolExecutor:
        # Double-checked locking: the common case (already created) reads
        # `_executor` lock-free; only the first caller(s) racing to create it
        # pay the lock, and only one of them actually constructs the pool —
        # otherwise concurrent first-requests could each build their own
        # "single" executor, defeating the serialization guarantee below.
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=len(self.retrievers))
        return self._executor

    def _get_enrichment_executor(self) -> ThreadPoolExecutor:
        if self._enrichment_executor is None:
            with self._enrichment_executor_lock:
                if self._enrichment_executor is None:
                    self._enrichment_executor = ThreadPoolExecutor(max_workers=1)
        return self._enrichment_executor

    def enrich_async(
        self, species: str, section: str, content: str, source_url: str = "", title: str = "",
    ):
        """
        Fire-and-forget write-back of a web-search fact into the persistent
        corpus, so the next time this species/topic comes up the answer is
        already in Supabase/Pinecone instead of paying for another Tavily call.

        No-op (returns None) if enrichment deps weren't wired in by init_rag()
        (e.g. Supabase/Pinecone unconfigured in local dev) — same graceful-
        degradation pattern used throughout this package. Otherwise returns
        the submitted Future; callers ignore it (truly fire-and-forget), but
        tests can call `.result()` on it to wait for the write deterministically.
        """
        if self.supabase_store is None or self.pinecone_index is None or self.embeddings is None:
            return None
        return self._get_enrichment_executor().submit(
            self._write_enrichment, species, section, content, source_url, title,
        )

    def _write_enrichment(
        self, species: str, section: str, content: str, source_url: str, title: str,
    ) -> None:
        try:
            species_id = self.supabase_store.get_species_id(species)
            if not species_id:
                log.info("Enrichment skipped — species not in curated corpus: %r", species)
                return

            # Reuses the existing delete-then-insert upsert_document (already
            # keyed on species_id/section/source) for idempotency — no new DB
            # constraint needed; a repeat write just replaces the prior scrape.
            self.supabase_store.upsert_document(
                species_id=species_id, section=section, content=content, source="web_enriched",
            )

            vector    = self.embeddings.embed_query(content)
            vector_id = f"web::{species}::{section}".lower().replace(" ", "_")
            self.pinecone_index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": vector,
                    "metadata": {
                        "species": species, "section": section, "source": "web_enriched",
                        "text": content[:1000], "url": source_url, "title": title,
                    },
                }],
                # Kept separate from the curated 'text' namespace — bulk-
                # purgeable without touching vetted data, since these facts
                # skip LLM/human verification before being written back.
                namespace="web_cache",
            )
            log.info("Enriched corpus: %s / %s", species, section)

            self._pending_enrichments += 1
            if self._pending_enrichments >= self.enrichment_rebuild_every:
                self._rebuild_bm25()
                self._pending_enrichments = 0
        except Exception as exc:
            log.warning("Enrichment write failed for %s / %s: %s", species, section, exc)

    def _rebuild_bm25(self) -> None:
        """
        Rebuild the BM25 sub-retriever from the full Supabase corpus (curated
        + enriched) and atomically swap it into `self.retrievers`, mirroring
        the lock-free cross-encoder hot-swap in factory.py's
        `_load_cross_encoder_async` — plain list-index assignment is
        GIL-atomic, so a concurrent `retrieve()` call sees either the old or
        the new BM25Retriever, never a half-built one. Debounced via
        `enrichment_rebuild_every` and run on the single-worker enrichment
        queue, so rebuilds never overlap or race each other.
        """
        try:
            docs = self.supabase_store.load_all_documents()
            if not docs:
                return
            new_bm25 = BM25Retriever.from_documents(docs, k=self._bm25_k())
            for i, retriever in enumerate(self.retrievers):
                if isinstance(retriever, BM25Retriever):
                    self.retrievers[i] = new_bm25
                    log.info("BM25 corpus refreshed — %d documents", len(docs))
                    return
        except Exception as exc:
            log.warning("BM25 rebuild failed: %s", exc)

    def _bm25_k(self) -> int:
        for retriever in self.retrievers:
            if isinstance(retriever, BM25Retriever):
                return retriever.k
        return 5

    def _fused_retrieve(
        self,
        query: str,
        species: str | None,
        web_cache: dict[str, list[Document]] | None = None,
    ) -> list[Document]:
        """
        Fuse local (BM25 + Pinecone) retrievers first; only fire the Tavily
        web retriever if the local result looks thin (see
        `_local_corpus_is_thin`) — Tavily is a metered network call and most
        queries are already answered by the curated corpus, so it shouldn't
        fire on every single retrieval.
        """
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

        def _accumulate(retrievers: list[Any], weights: list[float]) -> None:
            # Retrievers run concurrently (BM25 is in-memory but Pinecone/Tavily are
            # network calls) — futures are submitted and read back in `retrievers`
            # order, NOT completion order, so the "first retriever wins the stored
            # Document on collision" tie-break below stays deterministic regardless
            # of which network call happens to return first.
            pool = self._get_executor()
            futures = [pool.submit(_run, retriever) for retriever in retrievers]
            all_docs = [future.result() for future in futures]

            for docs, weight in zip(all_docs, weights):
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

        def _ranked() -> list[Document]:
            return [doc_map[k] for k in sorted(scores, key=scores.__getitem__, reverse=True)]

        local_pairs = [(r, w) for r, w in zip(self.retrievers, self.weights) if not isinstance(r, _TavilyRetriever)]
        web_pairs   = [(r, w) for r, w in zip(self.retrievers, self.weights) if isinstance(r, _TavilyRetriever)]

        _accumulate([r for r, _ in local_pairs], [w for _, w in local_pairs])

        is_thin, local_reranked = self._local_corpus_is_thin(query, _ranked())

        if web_pairs and is_thin:
            _accumulate([r for r, _ in web_pairs], [w for _, w in web_pairs])
            ranked = _ranked()
            if self.cross_encoder is not None:
                return self._rerank(query, ranked)
            return ranked[: self.final_k]

        # Web leg didn't fire — reuse the cross-encoder scoring already done
        # by the gating check above instead of re-running it on the same
        # candidates (cross-encoder inference is the most expensive step
        # here, and this is the common case: gating exists precisely so most
        # calls never need the web leg).
        if self.cross_encoder is not None:
            return local_reranked if local_reranked is not None else []
        return _ranked()[: self.final_k]

    def _local_corpus_is_thin(
        self, query: str, local_ranked: list[Document],
    ) -> tuple[bool, list[Document] | None]:
        """
        Gate for firing the (metered) web retriever: only worth paying for
        Tavily when BM25+Pinecone alone don't already have a good answer.

        RRF scores aren't a usable relevance signal for this — they're pure
        rank functions (`weight / (rrf_k + rank + 1)`), identical for a
        perfect match and a garbage match at the same rank. When the
        cross-encoder is loaded, it scores the actual (query, doc) pair, so
        reuse `rerank_threshold` — already tuned to mean "would this doc
        survive reranking at all" — as the gate. Before the cross-encoder
        finishes loading in the background (see factory.py's
        `_load_cross_encoder_async`), fall back to the cheap signal already
        used elsewhere in this class: zero local candidates at all.

        Returns (is_thin, local_reranked) — local_reranked is the already-
        cross-encoder-scored local-only list (None if the cross-encoder
        isn't loaded yet), so the caller can reuse it as the final result
        when the web leg doesn't fire, instead of reranking twice.
        """
        if not local_ranked:
            return True, None
        if self.cross_encoder is None:
            return False, None
        reranked = self._rerank(query, local_ranked)
        return (not reranked), reranked

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
