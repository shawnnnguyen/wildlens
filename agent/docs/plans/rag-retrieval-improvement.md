# Improve RAG Retrieval — `agent/src/safari_guide/rag.py`

## Context

The Safari Guide agent answers wildlife questions via a hybrid retriever (BM25 + Pinecone semantic, fused with Reciprocal Rank Fusion) defined in `rag.py` and invoked once per turn by `node_retrieve_information` in `nodes.py`. Live data was verified: 51 species, 102 document chunks in Supabase (BM25 corpus), 102 vectors in Pinecone's `text` namespace (in sync). That's a thin corpus (~2 chunks/species average).

The goal is to improve the RAG pipeline starting with retrieval. A deep review of `rag.py` + `node_retrieve_information` + the query construction feeding them (full code inspected, plus `data/fetcher.py` chunking, `data/supabase_store.py` read path, and `tests/test_rag.py`) surfaced concrete, code-grounded bugs and gaps — not generic RAG advice. This plan implements the prioritized fix list from that review, scoped strictly to the retrieval step (no ingestion/chunking rewrite, no new data sources).

Confirmed load-bearing fact: Gemini's `WildlifeIdentification.species` field is formatted `"Common name (Scientific name)"` (state.py's schema example: `"African Lion (Panthera leo)"`), and `species_list.json`'s `common_name` field stores exactly `"African Lion"` — so splitting on `"("` gives an exact match to the metadata `species` field stored in both Supabase and Pinecone. This makes species-metadata filtering safe to implement (with a soft fallback for the rare mismatch).

## Findings driving each change (file:function → concrete failure mode)

1. **No species filtering at retrieval time** (`node_retrieve_information`, nodes.py) — a follow-up like "what does it eat" after identifying a lion competes globally against all 51 species' chunks at k=5 per sub-retriever; the lion's own diet chunk can be crowded out. Biggest quality gap — this is the core Q&A loop.
2. **No re-rank / no relevance threshold / no final cap** (`_EnsembleRetriever._get_relevant_documents`, rag.py) — RRF scores are rank-based and always positive, so off-domain questions ("where's the wifi") still return animal chunks with positive scores, injected into the LLM prompt as "verified guidebook facts." Also can't distinguish a dense on-topic `characteristics` chunk from a long `overview` chunk that mentions the topic once.
3. **Bare `except Exception` around `retriever.invoke`** (rag.py:78-81) — a transient Pinecone network error routes into a fallback that calls the same network path again, raises again, and crashes the whole graph turn. Should degrade that one sub-retriever to `[]`, not crash or silently mask outages.
4. **`doc_map` keyed by `page_content[:120]`** (rag.py:84-88) — fragile cross-store identity. Currently "works" only by luck (Pinecone truncates stored text at 1000 chars in `ingest.py`, so the first 120 chars happen to match BM25's full content). Two different chunks sharing a 120-char prefix would silently corrupt scores (credit added to the wrong doc) — should key by the stable `(species, section, source)` metadata tuple instead (same fields `_pinecone_vector_id()` already uses at ingest).
5. **Naive `f"{species} {follow_up}".strip()` query** (nodes.py:203) — bakes the parenthesized scientific name into every query, which is off-distribution for the embedding model and can hijack BM25 toward name-mentioning chunks instead of the actual question intent.
6. **`k=5` per sub-retriever, no final cap** — miscalibrated once species-filtering lands (a species with more sections needs more candidates before fusion); the ensemble also returns everything fused (up to ~10 docs) with no cap on what gets injected into the persona prompt.

RRF itself is the right fusion primitive (rank-based, immune to BM25/cosine scale mismatch) — the fix is layering a cross-encoder re-rank on top, not replacing it. Sub-chunk splitting of long sections is a real precision issue but is an **ingestion-layer** concern, explicitly out of scope; the cross-encoder re-rank recovers most of that lost precision anyway since it attends over full chunk text.

## Implementation

All changes confined to `agent/src/safari_guide/rag.py` and `agent/src/safari_guide/nodes.py` (`node_retrieve_information`), plus corresponding additions to `agent/tests/test_rag.py`.

- [ ] **Step A — Harden `_EnsembleRetriever` fusion (rag.py)**
  In `_EnsembleRetriever._get_relevant_documents`:
  - Replace `key = doc.page_content[:120]` with a helper `_doc_key(doc)` returning `(metadata.get("species"), metadata.get("section"), metadata.get("source"))`, falling back to `hash(doc.page_content)` when all three metadata fields are empty (covers the mock corpus / `_MOCK_DOCUMENTS` case).
  - On a key collision, keep "first retriever wins" (BM25 is listed first in `init_rag`'s `retrievers=[bm25_retriever, pinecone_retriever]` and carries full untruncated content, so this also fixes the truncated-content issue as a side effect) — document why in a comment.
  - Replace the broad `try: docs = retriever.invoke(query) / except Exception: getattr(...)` with per-retriever isolation: wrap each sub-retriever's `.invoke(query)` call so any exception logs `log.warning(...)` and contributes `[]` for that source, instead of falling through to the deprecated `get_relevant_documents` method.
  - Add a final `[:final_k]` slice to the sorted output, with `final_k` as a new field on `_EnsembleRetriever` (default 6).

- [ ] **Step B — Retain the Pinecone vectorstore object (rag.py)**
  In `_init_pinecone_retriever`, return the `PineconeVectorStore` instance itself (or a small wrapper holding it + default `k`) instead of only the pre-bound `.as_retriever(...)`, so per-call filtered search (`vectorstore.similarity_search(query, k=k, filter={...})`) is possible in Step D. Keep `_NullRetriever` as the no-op stand-in when Pinecone is unavailable; keep the plain unfiltered path available for callers that don't pass a species filter.

- [ ] **Step C — Clean query construction (nodes.py)**
  In `node_retrieve_information`:
  - Parse `common_name = species.split("(")[0].strip()` from `state["identification_result"]["species"]`.
  - Build `query = f"{common_name} {follow_up}".strip() or "safari wildlife"` — drop the scientific name from the query string entirely.
  - Pass `common_name` through to the retriever as the species filter (Step D).

- [ ] **Step D — Species-filtered retrieval with soft fallback (rag.py + nodes.py)**
  - Add a public method `retrieve(query: str, species: str | None = None) -> list[Document]` on `_EnsembleRetriever` (keep `_get_relevant_documents` delegating to it with `species=None` so the class still satisfies `BaseRetriever`/existing tests).
    - Pinecone branch: if `species` is set and the retained vectorstore (Step B) is available, call `similarity_search(query, k=k, filter={"species": species})`; otherwise fall back to the existing unfiltered retriever/`_NullRetriever`.
    - BM25 branch: `BM25Retriever` has no metadata filter API — over-retrieve at a higher k (~15) then post-filter to `doc.metadata.get("species") == species` when `species` is set.
    - **Soft fallback:** if the species-filtered fusion yields zero documents, re-run once with `species=None` and use that result.
  - In `node_retrieve_information`, call `retriever.retrieve(query, species=common_name or None)` guarded by `hasattr(retriever, "retrieve")` so a plain `BaseRetriever` still works via `.invoke(query)`.

- [ ] **Step E — Cross-encoder re-rank + threshold (rag.py)**
  - In `init_rag`, load `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` once (CPU; `sentence-transformers` is already a project dependency) and pass it into `_EnsembleRetriever` as an optional field (`None` by default so existing tests that don't stub it still pass without the model download).
  - In `retrieve`, after RRF fusion, if a cross-encoder is configured: score `(query, doc.page_content)` for each fused candidate, sort by that score descending, drop candidates below a threshold, then slice to `final_k`. If everything falls below threshold, return `[]`.
  - Expose `final_k` and the threshold as `init_rag(...)` parameters with sensible defaults.

- [ ] **Tests (`agent/tests/test_rag.py`)**
  Add, using the existing `_TEST_CORPUS`/`_mock_init_rag` harness:
  - A doc returned by both sub-retrievers fuses into one entry under the new metadata-tuple key (not the old prefix key).
  - A metadata-key collision (two mock docs sharing content prefix but different `species`/`section`) no longer corrupts/drops either.
  - `retrieve(query, species="Lion")` returns only Lion-tagged docs from the mock corpus; `retrieve(query, species="Nonexistent")` (0 matches) falls back to the unfiltered result rather than returning empty.
  - An off-domain query against the cross-encoder-enabled path (when a cross-encoder is injected) returns `[]` once scores are below threshold — inject a stub scorer rather than downloading the real cross-encoder model, to keep tests fast/offline.

## Verification

1. `pytest agent/tests/test_rag.py -v` — all existing tests plus new ones pass, without requiring network/Pinecone/Supabase credentials (mock corpus + `_NullRetriever` path).
2. Manual live check using real credentials (repo-root `.env`): exercise `init_rag()` directly against live Supabase (102 docs) + Pinecone (102 vectors) from a scratch script, and confirm:
   - A species-filtered query for a known species (e.g. `"African Lion"`) returns only that species' chunks.
   - An unfiltered fallback still works when species is `None` or unmatched.
   - Off-domain query returns `[]` (or a short list) once the cross-encoder threshold is wired in.
3. Run the interactive CLI (`python -m safari_guide`, from `agent/`) for one photo-ID turn + one follow-up turn, confirm `retrieved_facts` in the logs (INFO level) is now scoped/capped and the persona script quality doesn't regress.
