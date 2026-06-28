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


class _EnsembleRetriever(BaseRetriever):
    """Minimal Reciprocal Rank Fusion retriever combining multiple sub-retrievers."""

    retrievers: list[Any]
    weights: list[float]
    rrf_k: int = 60

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        scores:  dict[str, float]    = {}
        doc_map: dict[str, Document] = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                docs = retriever.invoke(query)
            except Exception:
                docs = getattr(retriever, "get_relevant_documents", lambda q: [])(query)

            for rank, doc in enumerate(docs):
                key = doc.page_content[:120]
                if key not in doc_map:
                    doc_map[key] = doc
                    scores[key]  = 0.0
                scores[key] += weight / (self.rrf_k + rank + 1)

        return [doc_map[k] for k in sorted(scores, key=scores.__getitem__, reverse=True)]


def init_rag(
    k:               int   = 5,
    semantic_weight: float = 0.5,
    bm25_weight:     float = 0.5,
):
    """
    Return a hybrid EnsembleRetriever (BM25 + Pinecone semantic search).

    Args:
        k:               Number of documents each sub-retriever returns before fusion.
        semantic_weight: RRF weight for the Pinecone retriever (0–1).
        bm25_weight:     RRF weight for the BM25 retriever (0–1).

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

    # ── Hybrid fusion via Reciprocal Rank Fusion ──────────────────────────────
    log.info(
        f"Hybrid retriever ready — "
        f"BM25 weight={bm25_weight}, semantic weight={semantic_weight}, k={k}, "
        f"corpus={len(documents)} docs"
    )
    return _EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever],
        weights=[bm25_weight, semantic_weight],
        rrf_k=60,
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
        return vectorstore.as_retriever(search_kwargs={"k": k})

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
