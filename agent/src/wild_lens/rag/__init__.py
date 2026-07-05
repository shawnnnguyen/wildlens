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

Package layout
──────────────
ranking.py  — RRF fusion + cross-encoder re-ranking (_EnsembleRetriever)
backends.py — per-backend retriever adapters (Pinecone, Tavily, null stub)
factory.py  — init_rag() wiring: env vars, connections, graceful fallbacks
"""
from __future__ import annotations

from .backends import _NullRetriever, _PineconeRetrieverWrapper, _TavilyRetriever
from .factory import _MOCK_DOCUMENTS, init_rag
from .ranking import _doc_key, _EnsembleRetriever

__all__ = [
    "init_rag",
    "_EnsembleRetriever",
    "_NullRetriever",
    "_PineconeRetrieverWrapper",
    "_TavilyRetriever",
    "_MOCK_DOCUMENTS",
    "_doc_key",
]
