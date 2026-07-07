"""
Unit tests for RAG initialisation.
Uses mocked corpus and no-op retrievers to verify retrieval works end-to-end
without Pinecone, Supabase, or HuggingFace model download.
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from wildlens.rag import (
    _MOCK_DOCUMENTS,
    _EnsembleRetriever,
    _NullRetriever,
    _TavilyRetriever,
    _doc_key,
    init_rag,
)
from wildlens.rag.ranking import _bm25_search

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
        patch("wildlens.rag.factory.HuggingFaceEmbeddings", return_value=MagicMock()),
        patch("wildlens.rag.factory._init_pinecone_retriever", return_value=(_NullRetriever(), None)),
        patch("wildlens.rag.factory._load_bm25_corpus", return_value=(_TEST_CORPUS, None)),
        patch("wildlens.rag.factory._load_cross_encoder", return_value=None),
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

    Patches _local_corpus_is_thin to force the web leg to fire regardless of
    the gating heuristic (see test_gating_* below) — this test is about the
    RRF/doc_key fusion logic, not about when Tavily should be invoked.
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
    with patch.object(_EnsembleRetriever, "_local_corpus_is_thin", return_value=(True, None)):
        docs = retriever.invoke("lion facts")
    web_contents = {d.page_content for d in docs if d.metadata.get("source") == "web"}
    assert web_contents == {r["content"] for r in web_results}


def test_web_retriever_not_called_twice_on_species_fallback():
    """
    Tavily ignores species filtering, so the species-filtered pass and the
    unfiltered fallback pass in retrieve() must not double up the web call.

    Patches _local_corpus_is_thin to force both fusion passes to consider the
    web leg worth firing, so this test isolates the web_cache dedup logic
    from the gating heuristic (see test_gating_* below).
    """
    tavily, client = _make_tavily_stub([])
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)

    retriever = _EnsembleRetriever(retrievers=[bm25, _NullRetriever(), tavily], weights=[0.5, 0.3, 0.2])
    with patch.object(_EnsembleRetriever, "_local_corpus_is_thin", return_value=(True, None)):
        retriever.retrieve("some query", species="Nonexistent Species")

    assert client.search.call_count == 1


# ── Dynamic Search gating: only pay for Tavily when the local corpus is thin ─

def test_gating_skips_web_when_local_corpus_has_docs():
    """No cross-encoder loaded yet; BM25 already returns non-empty results for
    a well-matched query — Tavily must not be called at all."""
    tavily, client = _make_tavily_stub([{"content": "web fact", "url": "http://x", "title": "t"}])
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)

    retriever = _EnsembleRetriever(retrievers=[bm25, _NullRetriever(), tavily], weights=[0.5, 0.3, 0.2])
    retriever.invoke("lion facts")

    assert client.search.call_count == 0


def test_gating_fires_web_when_local_corpus_empty():
    """No local retrievers can answer at all — Tavily must fire."""
    tavily, client = _make_tavily_stub([{"content": "web fact", "url": "http://x", "title": "t"}])

    retriever = _EnsembleRetriever(retrievers=[_NullRetriever(), tavily], weights=[0.5, 0.2])
    retriever.invoke("some obscure query")

    assert client.search.call_count == 1


def test_gating_uses_cross_encoder_score_not_rrf_rank():
    """
    RRF scores alone can't signal relevance (pure rank function) — once the
    cross-encoder is loaded, gating must key off its (query, doc) score
    instead. Local docs exist and rank fine in RRF, but a cross-encoder that
    scores everything below rerank_threshold means the local corpus is
    genuinely a poor match, so Tavily should still fire.
    """
    tavily, client = _make_tavily_stub([{"content": "web fact", "url": "http://x", "title": "t"}])
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)
    fake_ce = MagicMock()
    fake_ce.predict.side_effect = lambda pairs: [-5.0] * len(pairs)

    retriever = _EnsembleRetriever(
        retrievers=[bm25, _NullRetriever(), tavily], weights=[0.5, 0.3, 0.2],
        cross_encoder=fake_ce, rerank_threshold=0.0,
    )
    retriever.invoke("lion facts")

    assert client.search.call_count == 1


# ── Knowledge Base Enrichment: web facts written back to Supabase/Pinecone ──

def test_enrich_async_noop_without_deps():
    """enrich_async must no-op (not raise) when Supabase/Pinecone/embeddings
    weren't wired in — e.g. local dev without those services configured."""
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)
    retriever = _EnsembleRetriever(retrievers=[bm25], weights=[1.0])

    result = retriever.enrich_async(species="Lion", section="diet", content="Lions eat meat")

    assert result is None
    assert retriever._enrichment_executor is None


def test_enrich_async_writes_to_supabase_and_pinecone():
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)
    store = MagicMock()
    store.get_species_id.return_value = 1
    pinecone_index = MagicMock()
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.0] * 384

    retriever = _EnsembleRetriever(
        retrievers=[bm25], weights=[1.0],
        supabase_store=store, pinecone_index=pinecone_index, embeddings=embeddings,
    )

    retriever.enrich_async(species="Lion", section="diet", content="Lions eat meat").result()

    store.upsert_document.assert_called_once_with(
        species_id=1, section="diet", content="Lions eat meat", source="web_enriched",
    )
    pinecone_index.upsert.assert_called_once()
    assert pinecone_index.upsert.call_args.kwargs["namespace"] == "web_cache"


def test_enrich_async_skips_write_for_unknown_species():
    """Never persist an enrichment fact for a species not already in the
    curated corpus — there's no species_id to attach it to."""
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)
    store = MagicMock()
    store.get_species_id.return_value = None
    pinecone_index = MagicMock()
    embeddings = MagicMock()

    retriever = _EnsembleRetriever(
        retrievers=[bm25], weights=[1.0],
        supabase_store=store, pinecone_index=pinecone_index, embeddings=embeddings,
    )

    retriever.enrich_async(species="Unknown Critter", section="diet", content="...").result()

    store.upsert_document.assert_not_called()
    pinecone_index.upsert.assert_not_called()


def test_enrich_async_debounces_bm25_rebuild():
    """BM25 is rebuilt-and-swapped only after enrichment_rebuild_every writes,
    not on every single enrichment — avoids rebuilding the corpus per-call."""
    bm25 = BM25Retriever.from_documents(_TEST_CORPUS, k=5)
    store = MagicMock()
    store.get_species_id.return_value = 1
    store.load_all_documents.return_value = _TEST_CORPUS
    pinecone_index = MagicMock()
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.0] * 384

    retriever = _EnsembleRetriever(
        retrievers=[bm25], weights=[1.0],
        supabase_store=store, pinecone_index=pinecone_index, embeddings=embeddings,
        enrichment_rebuild_every=2,
    )

    retriever.enrich_async(species="Lion", section="diet", content="fact one").result()
    assert store.load_all_documents.call_count == 0
    assert retriever.retrievers[0] is bm25

    retriever.enrich_async(species="Lion", section="habitat", content="fact two").result()
    assert store.load_all_documents.call_count == 1
    assert isinstance(retriever.retrievers[0], BM25Retriever)
    assert retriever.retrievers[0] is not bm25  # atomically swapped for a fresh instance


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
        patch("wildlens.rag.factory.HuggingFaceEmbeddings", return_value=MagicMock()),
        patch("wildlens.rag.factory._init_pinecone_retriever", return_value=(_NullRetriever(), None)),
        patch("wildlens.rag.factory._load_bm25_corpus", return_value=(_TEST_CORPUS, None)),
    ):
        return init_rag(**kwargs)


def test_init_rag_returns_immediately_with_reranker_pending():
    """cross_encoder starts None (RRF-only) — the background thread is not
    forced to complete before init_rag() returns, matching non-blocking startup."""
    with patch("wildlens.rag.factory.threading.Thread") as mock_thread:
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
        patch("wildlens.rag.factory.threading.Thread.start", _run_synchronously, create=True),
        patch("wildlens.rag.factory._load_cross_encoder", return_value=fake_model),
    ):
        retriever = _mock_init_rag_no_reranker_patch(use_reranker=True)

    assert retriever.cross_encoder is fake_model


# ── Tavily daily call cap ─────────────────────────────────────────────────────

def test_tavily_daily_cap_blocks_after_limit():
    client = MagicMock()
    client.search.return_value = {"results": [{"content": "x", "url": "http://x", "title": "t"}]}
    tavily = _TavilyRetriever(client=client, k=1)

    with patch.dict(os.environ, {"TAVILY_DAILY_CALL_CAP": "2"}):
        tavily.invoke("q1")
        tavily.invoke("q2")
        docs = tavily.invoke("q3")

    assert client.search.call_count == 2
    assert docs == []


def test_tavily_daily_cap_resets_next_day():
    client = MagicMock()
    client.search.return_value = {"results": [{"content": "x", "url": "http://x", "title": "t"}]}
    tavily = _TavilyRetriever(client=client, k=1)

    with patch.dict(os.environ, {"TAVILY_DAILY_CALL_CAP": "1"}):
        tavily.invoke("q1")
        assert tavily.invoke("q2") == []  # cap hit for today

        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        with patch("wildlens.rag.backends.datetime") as mock_dt:
            mock_dt.date.today.return_value = tomorrow
            tavily.invoke("q3")

    assert client.search.call_count == 2


def test_tavily_daily_cap_malformed_env_var_falls_back_to_default():
    """A non-integer TAVILY_DAILY_CALL_CAP must not crash retrieval — fall
    back to the default cap instead of raising ValueError on every call."""
    client = MagicMock()
    client.search.return_value = {"results": []}
    tavily = _TavilyRetriever(client=client, k=1)

    with patch.dict(os.environ, {"TAVILY_DAILY_CALL_CAP": "not-a-number"}):
        tavily.invoke("q1")  # must not raise

    assert client.search.call_count == 1


# ── web_cache Pinecone namespace wiring ───────────────────────────────────────

def test_init_web_cache_retriever_returns_none_without_pinecone_index():
    """No Pinecone index means no namespace to build a web_cache retriever on
    top of — must degrade to None (excluded from the ensemble) rather than error."""
    from wildlens.rag.factory import _init_web_cache_retriever

    assert _init_web_cache_retriever(MagicMock(), None, k=5) is None


def test_init_web_cache_retriever_wraps_web_cache_namespace():
    from wildlens.rag import _PineconeRetrieverWrapper
    from wildlens.rag.factory import _init_web_cache_retriever

    fake_index = MagicMock()
    with patch("langchain_pinecone.PineconeVectorStore") as mock_vs:
        mock_vs.return_value = MagicMock()
        retriever = _init_web_cache_retriever(MagicMock(), fake_index, k=5)

    assert isinstance(retriever, _PineconeRetrieverWrapper)
    mock_vs.assert_called_once()
    assert mock_vs.call_args.kwargs["namespace"] == "web_cache"
