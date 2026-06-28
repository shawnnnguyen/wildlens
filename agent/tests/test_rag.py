"""
Unit tests for RAG initialisation.
Uses mocked corpus and no-op retrievers to verify retrieval works end-to-end
without Pinecone, Supabase, or HuggingFace model download.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from safari_guide.rag import _MOCK_DOCUMENTS, _EnsembleRetriever, _NullRetriever, init_rag

_TEST_CORPUS = [
    Document(page_content="The African lion is an apex predator of the savanna.", metadata={"species": "Lion"}),
    Document(page_content="Hippos are highly dangerous near water and responsible for many attacks.", metadata={"species": "Hippo"}),
    Document(page_content="Zebras live in large herds on the plains of East Africa.", metadata={"species": "Zebra"}),
    Document(page_content="Elephants are the largest land mammals and display complex social behaviour.", metadata={"species": "Elephant"}),
    Document(page_content="Leopards are solitary big cats that hunt at night in the Serengeti.", metadata={"species": "Leopard"}),
]


def _mock_init_rag(**kwargs):
    """Run init_rag() with all external calls stubbed out."""
    with (
        patch("safari_guide.rag.HuggingFaceEmbeddings", return_value=MagicMock()),
        patch("safari_guide.rag._init_pinecone_retriever", return_value=_NullRetriever()),
        patch("safari_guide.rag._load_bm25_corpus", return_value=_TEST_CORPUS),
    ):
        return init_rag(**kwargs)


def test_mock_documents_are_nonempty():
    assert len(_MOCK_DOCUMENTS) >= 1


def test_init_rag_returns_ensemble_retriever():
    retriever = _mock_init_rag()
    assert isinstance(retriever, _EnsembleRetriever)


def test_retrieval_returns_relevant_doc():
    retriever = _mock_init_rag()
    docs = retriever.invoke("African lion hunting behaviour")
    assert docs
    assert any("lion" in d.page_content.lower() for d in docs)


def test_retrieval_for_safety_query():
    retriever = _mock_init_rag()
    docs = retriever.invoke("hippo safety danger water")
    assert any("hippo" in d.page_content.lower() for d in docs)


def test_retrieval_degrades_to_bm25_without_pinecone():
    """BM25-only retrieval still works when Pinecone is unavailable."""
    retriever = _mock_init_rag()
    docs = retriever.invoke("zebra plains herd")
    assert docs
