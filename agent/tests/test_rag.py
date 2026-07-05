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

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from wild_lens.rag import (
    _MOCK_DOCUMENTS,
    _EnsembleRetriever,
    _NullRetriever,
    _TavilyRetriever,
    _doc_key,
    init_rag,
)

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
        patch("wild_lens.rag.factory.HuggingFaceEmbeddings", return_value=MagicMock()),
        patch("wild_lens.rag.factory._init_pinecone_retriever", return_value=_NullRetriever()),
        patch("wild_lens.rag.factory._load_bm25_corpus", return_value=_TEST_CORPUS),
        patch("wild_lens.rag.factory._load_cross_encoder", return_value=None),
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


def test_doc_key_unique_per_web_url():
    """Web docs must not collide in _doc_key just because they share source='web'."""
    doc1 = Document(page_content="a", metadata={"source": "web", "url": "http://x/1"})
    doc2 = Document(page_content="b", metadata={"source": "web", "url": "http://x/2"})
    assert _doc_key(doc1) != _doc_key(doc2)


def _make_tavily_stub(results: list[dict]) -> tuple[_TavilyRetriever, MagicMock]:
    client = MagicMock()
    client.search.return_value = {"results": results}
    return _TavilyRetriever(client=client, k=len(results) or 1), client


def test_ensemble_retrieve_keeps_multiple_web_docs():
    """
    Regression test: every Tavily Document shares (species=None, section=None,
    source='web'), which used to collapse all web hits from one query into a
    single fused entry and silently drop the rest.
    """
    web_results = [
        {"content": f"web fact {i}", "url": f"http://x/{i}", "title": f"Page {i}"}
        for i in range(3)
    ]
    tavily, _ = _make_tavily_stub(web_results)
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)

    retriever = _EnsembleRetriever(
        retrievers=[bm25, _NullRetriever(), tavily],
        weights=[0.5, 0.3, 0.2],
        final_k=10,
    )
    docs = retriever.invoke("lion facts")
    web_contents = {d.page_content for d in docs if d.metadata.get("source") == "web"}
    assert web_contents == {r["content"] for r in web_results}


def test_web_retriever_not_called_twice_on_species_fallback():
    """
    Tavily ignores species filtering, so the species-filtered pass and the
    unfiltered fallback pass in retrieve() must not double up the web call.
    """
    tavily, client = _make_tavily_stub([])
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)

    retriever = _EnsembleRetriever(retrievers=[bm25, _NullRetriever(), tavily], weights=[0.5, 0.3, 0.2])
    retriever.retrieve("some query", species="Nonexistent Species")

    assert client.search.call_count == 1
