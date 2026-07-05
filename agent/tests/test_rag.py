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
from wild_lens.rag.ranking import _bm25_search

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


# ── Bug #8: ThreadPoolExecutor is created once and reused ───────────────────

def test_executor_is_reused_across_retrieve_calls():
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)
    retriever = _EnsembleRetriever(retrievers=[bm25], weights=[1.0])

    retriever.invoke("lion")
    first_executor = retriever._get_executor()
    retriever.invoke("hippo")
    second_executor = retriever._get_executor()

    assert first_executor is second_executor


# ── Bug #6: BM25 internals fallback + canary ─────────────────────────────────

def test_bm25_internal_api_still_present():
    """
    Canary: if this fails, langchain_community changed BM25Retriever internals —
    update ranking.py's _bm25_search fast path (and its try/except fallback)
    accordingly.
    """
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=3)
    assert hasattr(bm25, "preprocess_func")
    assert hasattr(bm25, "vectorizer") and hasattr(bm25.vectorizer, "get_top_n")
    assert hasattr(bm25, "docs")


def test_bm25_search_falls_back_when_internals_missing():
    """
    Simulates a langchain_community upgrade that renames/removes the private
    internals _bm25_search relies on, while the public .invoke() API keeps
    working (using whatever new internals the library switched to) — the
    fallback must use that public path instead of crashing.
    """
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)
    canned_docs = [d for d in _TEST_CORPUS if d.metadata["species"] == "Lion"]
    with (
        patch.object(bm25, "vectorizer", object()),  # no get_top_n -> AttributeError
        patch.object(BM25Retriever, "_get_relevant_documents", return_value=canned_docs),
    ):
        docs = _bm25_search(bm25, "lion", n=5, species="Lion")
    assert docs == canned_docs


# ── Bug #9: background cross-encoder loading ─────────────────────────────────

def _mock_init_rag_no_reranker_patch(**kwargs):
    """Like _mock_init_rag but leaves _load_cross_encoder unpatched, so callers
    can control it themselves for background-loading tests."""
    with (
        patch("wild_lens.rag.factory.HuggingFaceEmbeddings", return_value=MagicMock()),
        patch("wild_lens.rag.factory._init_pinecone_retriever", return_value=_NullRetriever()),
        patch("wild_lens.rag.factory._load_bm25_corpus", return_value=_TEST_CORPUS),
    ):
        return init_rag(**kwargs)


def test_init_rag_returns_immediately_with_reranker_pending():
    """cross_encoder starts None (RRF-only) — the background thread is not
    forced to complete before init_rag() returns, matching non-blocking startup."""
    with patch("wild_lens.rag.factory.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()  # .start() is a no-op, never runs the target
        retriever = _mock_init_rag_no_reranker_patch(use_reranker=True)
    assert retriever.cross_encoder is None
    mock_thread.assert_called_once()


def test_cross_encoder_becomes_active_after_background_load_completes():
    """With the background thread forced to run synchronously, cross_encoder
    becomes non-None once the (mocked) load finishes."""
    fake_model = MagicMock()

    def _run_synchronously(self):
        self._target(*self._args, **self._kwargs)

    with (
        patch("wild_lens.rag.factory.threading.Thread.start", _run_synchronously, create=True),
        patch("wild_lens.rag.factory._load_cross_encoder", return_value=fake_model),
    ):
        retriever = _mock_init_rag_no_reranker_patch(use_reranker=True)

    assert retriever.cross_encoder is fake_model
