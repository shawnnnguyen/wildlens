"""
RAG initialisation: hybrid retriever combining Pinecone semantic search + BM25 keyword search.

Retrieval strategy
──────────────────
Semantic (Pinecone) — cloud vector store; 384-dim all-MiniLM-L6-v2 embeddings;
                       catches paraphrases and concept-level matches.
Keyword  (BM25)     — rebuilt in-memory from Supabase document corpus each startup;
                       catches exact species names and rare terms.
Fusion              — EnsembleRetriever merges both ranked lists via Reciprocal Rank
                       Fusion (RRF), giving each document a combined score.

Data sources
────────────
Pinecone (namespace='text') — pre-ingested EOL + IUCN + handcrafted document vectors.
Supabase (documents table)  — raw document chunks loaded for BM25 rebuild at startup.

Fallback
────────
If Supabase is unreachable or returns no documents, BM25 falls back to the
built-in mock corpus (_MOCK_DOCUMENTS) so the app is never completely broken
even without a live database connection.

Ingestion
─────────
Run once before first use:
    python -m safari_guide.data.ingest --text
    python -m safari_guide.data.ingest --images   # optional
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[assignment]

log = logging.getLogger("safari_guide.rag")


# Emergency fallback — used only if Supabase is unreachable at startup
_MOCK_DOCUMENTS: list[Document] = [
    Document(
        page_content="No documents loaded. Run: python -m safari_guide.data.ingest --text",
        metadata={"species": "Unknown", "section": "error", "threat_level": "low"},
    )
]


def _doc_key(doc: Document) -> tuple[str | None, str | None, str | None] | int:
    """
    Stable cross-retriever identity for a document, used to fuse duplicate
    hits from BM25 and Pinecone into a single scored entry.

    Keyed on the same (species, section, source) triple used at ingest time
    (see `_pinecone_vector_id()`), rather than a content prefix — two distinct
    chunks that happen to share their first N characters must not collide.
    Falls back to a content hash only for documents with no identifying
    metadata at all (e.g. the `_MOCK_DOCUMENTS` corpus).
    """
    metadata = doc.metadata or {}
    species = metadata.get("species")
    section = metadata.get("section")
    source  = metadata.get("source")
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

    def _fused_retrieve(self, query: str, species: str | None) -> list[Document]:
        scores:  dict[Any, float]    = {}
        doc_map: dict[Any, Document] = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                if species and hasattr(retriever, "similarity_search"):
                    docs = retriever.similarity_search(query, filter={"species": species})
                elif species and isinstance(retriever, BM25Retriever):
                    docs = _bm25_search(retriever, query, n=15, species=species)
                else:
                    docs = retriever.invoke(query)
            except Exception as exc:
                log.warning(f"Retriever {retriever!r} failed for query {query!r}: {exc}")
                docs = []

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
        docs = self._fused_retrieve(query, species)
        if species is not None and not docs:
            log.info(f"No docs for species={species!r} — falling back to unfiltered retrieval")
            docs = self._fused_retrieve(query, None)
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.retrieve(query, species=None)


def init_rag(
    k:                int   = 5,
    semantic_weight:  float = 0.5,
    bm25_weight:      float = 0.5,
    final_k:          int   = 6,
    rerank_threshold: float = 0.0,
    use_reranker:     bool  = True,
):
    """
    Return a hybrid EnsembleRetriever (BM25 + Pinecone semantic search).

    Args:
        k:                Number of documents each sub-retriever returns before fusion.
        semantic_weight:  RRF weight for the Pinecone retriever (0–1).
        bm25_weight:      RRF weight for the BM25 retriever (0–1).
        final_k:          Max documents returned after fusion (and re-ranking, if enabled).
        rerank_threshold: Minimum cross-encoder relevance score to keep a candidate.
        use_reranker:     Load a cross-encoder to re-rank fused candidates.

    Env vars consumed:
        PINECONE_API_KEY, PINECONE_INDEX_NAME  — Pinecone connection
        SUPABASE_URL, SUPABASE_KEY             — Supabase connection for BM25 corpus
    """
    log.info("Loading HuggingFace embedding model (all-MiniLM-L6-v2) …")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Pinecone semantic retriever ───────────────────────────────────────────
    pinecone_retriever = _init_pinecone_retriever(embeddings, k)

    # ── BM25 keyword retriever — corpus loaded from Supabase ─────────────────
    documents = _load_bm25_corpus()
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    # ── Cross-encoder re-rank (optional) ──────────────────────────────────────
    cross_encoder = (
        _load_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None
    )

    # ── Hybrid fusion via Reciprocal Rank Fusion ──────────────────────────────
    log.info(
        f"Hybrid retriever ready — "
        f"BM25 weight={bm25_weight}, semantic weight={semantic_weight}, k={k}, "
        f"corpus={len(documents)} docs, reranker={'on' if cross_encoder else 'off'}"
    )
    return _EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever],
        weights=[bm25_weight, semantic_weight],
        rrf_k=60,
        final_k=final_k,
        cross_encoder=cross_encoder,
        rerank_threshold=rerank_threshold,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _init_pinecone_retriever(embeddings, k: int):
    """Connect to Pinecone and return a retriever over the 'text' namespace."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name       = os.getenv("PINECONE_INDEX_NAME", "safari-guide")

    if not pinecone_api_key:
        log.warning(
            "PINECONE_API_KEY not set — semantic retriever will be a no-op stub. "
            "Run: python -m safari_guide.data.ingest --text to populate."
        )
        return _NullRetriever(k=k)

    try:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        pc       = Pinecone(api_key=pinecone_api_key)
        index    = pc.Index(index_name)
        stats    = index.describe_index_stats()
        ns_count = stats.get("namespaces", {}).get("text", {}).get("vector_count", 0)

        log.info(f"Pinecone index '{index_name}' connected — {ns_count} vectors in 'text' namespace")

        if ns_count == 0:
            log.warning(
                "Pinecone 'text' namespace is empty. "
                "Run: python -m safari_guide.data.ingest --text"
            )

        vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace="text")
        return _PineconeRetrieverWrapper(vectorstore=vectorstore, k=k)

    except Exception as exc:
        log.warning(f"Pinecone init failed ({exc}) — falling back to null semantic retriever")
        return _NullRetriever(k=k)


def _load_bm25_corpus() -> list[Document]:
    """
    Load document chunks from Supabase for BM25 rebuild.
    Falls back to _MOCK_DOCUMENTS if Supabase is unreachable or empty.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        log.warning(
            "SUPABASE_URL or SUPABASE_KEY not set — BM25 using mock corpus. "
            "Set credentials and run: python -m safari_guide.data.ingest --text"
        )
        return _MOCK_DOCUMENTS

    try:
        from .data.supabase_store import SupabaseStore
        store = SupabaseStore()
        docs  = store.load_all_documents()
        if docs:
            return docs
        log.warning(
            "Supabase documents table is empty — BM25 using mock corpus. "
            "Run: python -m safari_guide.data.ingest --text"
        )
        return _MOCK_DOCUMENTS
    except Exception as exc:
        log.warning(f"Supabase load failed ({exc}) — BM25 using mock corpus")
        return _MOCK_DOCUMENTS


def _load_cross_encoder(model_name: str) -> Any | None:
    """
    Load a cross-encoder for re-ranking fused RRF candidates.
    Returns None on any failure (offline environment, no cached weights,
    missing dependency) so _EnsembleRetriever degrades to RRF-only ranking —
    the same graceful-degradation pattern used for Pinecone/Supabase above.
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_name)
        log.info(f"Cross-encoder '{model_name}' loaded for re-ranking")
        return model
    except Exception as exc:
        log.warning(f"Cross-encoder load failed ({exc}) — re-ranking disabled, using RRF scores only")
        return None


# ── Pinecone retriever wrapper ────────────────────────────────────────────────

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


# ── Null retriever stub ───────────────────────────────────────────────────────

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
