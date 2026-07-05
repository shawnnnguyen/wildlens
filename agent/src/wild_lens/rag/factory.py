"""
init_rag(): builds the hybrid retriever stack from config/env vars.

Connects to Pinecone, Supabase, and Tavily, and falls back gracefully
(null retriever / mock corpus / no reranker) when any of them is
unreachable or unconfigured, so the app is never completely broken.

Ingestion
─────────
Run once before first use:
    python -m safari_guide.data.ingest --text
    python -m safari_guide.data.ingest --images   # optional
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[assignment]

from .backends import _NullRetriever, _PineconeRetrieverWrapper, _TavilyRetriever
from .ranking import _EnsembleRetriever

log = logging.getLogger(__name__)


# Emergency fallback — used only if Supabase is unreachable at startup
_MOCK_DOCUMENTS: list[Document] = [
    Document(
        page_content="No documents loaded. Run: python -m safari_guide.data.ingest --text",
        metadata={"species": "Unknown", "section": "error", "threat_level": "low"},
    )
]


def init_rag(
    k:                int   = 5,
    semantic_weight:  float = 0.5,
    bm25_weight:      float = 0.5,
    web_weight:       float = 0.2,
    final_k:          int   = 6,
    rerank_threshold: float = 0.0,
    use_reranker:     bool  = True,
):
    """
    Return a hybrid EnsembleRetriever (BM25 + Pinecone semantic search + Tavily web search).

    Args:
        k:                Number of documents each sub-retriever returns before fusion.
        semantic_weight:  RRF weight for the Pinecone retriever (0–1).
        bm25_weight:      RRF weight for the BM25 retriever (0–1).
        web_weight:       RRF weight for the Tavily web retriever (0–1). Kept lower than
                           the internal sources by default — web results are unvetted and
                           only meant to supplement the curated guidebook corpus.
        final_k:          Max documents returned after fusion (and re-ranking, if enabled).
        rerank_threshold: Minimum cross-encoder relevance score to keep a candidate.
        use_reranker:     Load a cross-encoder to re-rank fused candidates.

    Env vars consumed:
        PINECONE_API_KEY, PINECONE_INDEX_NAME  — Pinecone connection
        SUPABASE_URL, SUPABASE_KEY             — Supabase connection for BM25 corpus
        TAVILY_API_KEY                         — Tavily web search connection
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

    # ── Tavily web search retriever ───────────────────────────────────────────
    tavily_retriever = _init_tavily_retriever(k)

    # ── Hybrid fusion via Reciprocal Rank Fusion ──────────────────────────────
    log.info(
        f"Hybrid retriever ready — "
        f"BM25 weight={bm25_weight}, semantic weight={semantic_weight}, web weight={web_weight}, "
        f"k={k}, corpus={len(documents)} docs"
    )
    retriever = _EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever, tavily_retriever],
        weights=[bm25_weight, semantic_weight, web_weight],
        rrf_k=60,
        final_k=final_k,
        cross_encoder=None,   # loaded in the background below; RRF-only until ready
        rerank_threshold=rerank_threshold,
    )

    # ── Cross-encoder re-rank (optional, loaded in the background) ───────────
    # Loading synchronously here would block the entire app's startup/readiness
    # on downloading model weights from HuggingFace Hub (see backend/main.py's
    # lifespan(), which awaits init_rag() before the app is considered ready).
    # Requests served before this finishes simply get RRF-only ranking — no
    # worse than the existing "reranker disabled" fallback path already used
    # when the load fails outright.
    if use_reranker:
        threading.Thread(
            target=_load_cross_encoder_async,
            args=(retriever, "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            daemon=True,
        ).start()

    return retriever


def _load_cross_encoder_async(retriever: _EnsembleRetriever, model_name: str) -> None:
    """Background-thread target: load the cross-encoder, then attach it to the
    already-returned retriever. Plain attribute assignment — _EnsembleRetriever
    doesn't set validate_assignment, so this is a cheap, GIL-atomic swap."""
    model = _load_cross_encoder(model_name)
    retriever.cross_encoder = model
    if model is not None:
        log.info("Cross-encoder ready — reranking now active for subsequent requests")


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


def _init_tavily_retriever(k: int):
    """Connect to Tavily and return a web-search retriever, or a no-op stub."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        log.warning(
            "TAVILY_API_KEY not set — web retriever will be a no-op stub. "
            "Set the key to enable live web search fallback."
        )
        return _NullRetriever(k=k)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_api_key)
        log.info("Tavily web search client connected")
        return _TavilyRetriever(client=client, k=k)

    except Exception as exc:
        log.warning(f"Tavily init failed ({exc}) — falling back to null web retriever")
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
        from ..data.supabase_store import SupabaseStore
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
