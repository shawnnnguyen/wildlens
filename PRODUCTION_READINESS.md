# Wildlens — Production Readiness Roadmap

Living tracker for hardening `wildlens` (LangGraph agent in `agent/src/wildlens/` + FastAPI backend in `backend/`) toward production. Originated from an architecture review against four pillars: Architecture & Orchestration, Reliability & Guardrails, Cost & Latency Optimization, Observability & Evaluation. Update this file as items land — check them off, don't delete the history.

**Product context that shapes Phase 1 below:** this is a single-use, accountless app — tourists snap a photo, chat, and leave; nobody expects the session to survive after they close the tab. That rules out a login/user-account model as the fix for session security, and it means the eval/observability gap should build on the Langfuse integration already in place rather than a parallel bespoke harness. See the revised Phase 1 auth item and Phase 3 eval item below.

**Last updated:** 2026-07-07

## Executive Summary

Wildlens is a well-architected **prototype**: the LangGraph state machine is clean, reducers are used correctly, the RAG ensemble is genuinely hybrid (BM25 + Pinecone + Tavily with RRF fusion), and Langfuse tracing is *actually wired in* (not just documented) with custom image redaction. This is above-average engineering discipline for a project at this stage.

It is **not yet production-hardened**. Highest-priority risks, in order:

1. **Unbounded/leaking memory** — `identification_history` grows forever per session; `DELETE /api/sessions/{id}` doesn't actually clear the in-memory checkpointer, only a tracking set. Combined with `MemorySaver` (pure in-process RAM, no persistence), this is a slow memory leak and a full data-loss-on-restart risk.
2. **No per-session secret** — `thread_id` is a client-generated UUID (`crypto.randomUUID()`, `frontend/src/hooks/useSessions.ts:8,56`) that doubles as the *only* credential for chat continuation, `GET /history`, and `DELETE`. Anyone who obtains the ID (browser history, a shared screen, a proxy log) can read or evict someone else's session. This is not a missing-auth-model problem — the app is intentionally accountless and single-use — it's a missing capability token, a much smaller fix.
3. **Silent failure modes** — two of the three LLM-calling nodes have zero error handling; the RAG ensemble swallows sub-retriever failures into an empty result set with no alerting. No retry/backoff anywhere in the codebase.
4. **No accuracy evaluation** — for an app whose entire value proposition is "correctly identify this animal," there is no golden dataset, no LLM-as-judge, no regression tracking.
5. **No cost controls** — no semantic caching, no `max_tokens` caps, single model tier for cheap and expensive tasks alike.

None of this requires a rebuild — the graph topology, state schema, and RAG layer are solid foundations. The gaps are additive (retry decorators, an auth check, a caching layer, an eval harness), not structural.

---

## Pillar 1: Architecture & Orchestration

**Current state**
- Single fixed `StateGraph` (`agent/src/wildlens/graphs.py`), compiled once with `MemorySaver()` (line 235 at time of writing). Routing is three binary conditional edges (`route_entry`, `route_after_analysis`, `route_audio`) plus, as of PR #9, a topic-relevance gate — see Phase 2 below.
- `node_generate_guide_persona` (`nodes.py:601-704`) is a monolithic call: persona narration + fact reconciliation (guidebook vs. web) + cross-session digest + intro-vs-follow-up branching, all in one `llm.invoke()`.
- `chat_history` summarization is well-implemented: sliding 6-message raw window, delta-only LLM calls beyond `SUMMARY_THRESHOLD=10` (`nodes.py:538-594`).
- `identification_history` (`state.py:66`, `operator.add` reducer) is **unbounded**.
- No cross-session/long-term vector memory; `MemorySaver` is pure in-process RAM (no persistence across restarts, no horizontal scaling).

---

## Pillar 2: Reliability & Guardrails

**Current state**
- `node_analyze_image` has a graceful try/except fallback; `node_summarize_history` and `node_generate_guide_persona` have **no error handling** — failures bubble to a generic 500.
- No retry/backoff library anywhere (no `tenacity`, no manual backoff) for LLM calls, Pinecone, Supabase, or ingestion HTTP clients.
- TTS fallback chain (`tts.py`) only covers *import-time* unavailability, not *runtime* failures — a network drop mid-synthesis raises uncaught instead of falling through edge-tts → gTTS.
- No fallback LLM provider for Gemini (vision) or DeepSeek (text) — single point of failure per modality.
- Supabase runtime reads use the same service-role key as ingestion writes — no read-only credential split.
- `DELETE /api/sessions/{thread_id}` (and `GET /history`, and chat continuation) trust a client-generated `thread_id` as if it were a secret — nothing is checked against it, so knowing/guessing a `thread_id` is sufficient to read or evict someone else's session. `DELETE` also doesn't actually clear the `MemorySaver` checkpoint (only a tracking set) — unauthorized *and* incomplete.
- No human-in-the-loop gate anywhere in the graph.
- Global exception handler does *not* leak stack traces to clients — this part is solid.

---

## Pillar 3: Cost & Latency Optimization

**Current state**
- Single model tier per modality: `gemini-2.5-flash` for vision (`backend/main.py:64`, `GEMINI_VISION_MODEL`), `deepseek-chat` for all text generation (summarization *and* persona generation reuse the same instance, no cheaper tier for the lighter task). No `max_tokens` cap set anywhere.
- No semantic/response caching anywhere in the runtime path (repo-wide grep for redis/diskcache/gptcache/lru_cache returns nothing in production code).
- ~~Retrieval fired unconditionally on every text turn, including small talk~~ — **addressed by PR #9**, see Phase 2.

---

## Pillar 4: Observability & Evaluation

**Current state**
- Real Langfuse integration (`agent/src/wildlens/observability.py`) — custom image redaction (base64 data URIs stripped, long strings truncated; no actual PII/name/email scrubbing despite the "PII redaction" label this used to carry in this doc), per-turn spans, manual `@observe()` wrapping for plain-Python nodes, and per-turn metadata (species/confidence/threat_level) already attached to every trace (`observability.py:96-104`). Genuinely solid, not just documented env vars. Token usage/cost per generation is also already captured natively by Langfuse's `CallbackHandler` — no extra aggregation layer needed, just query it.
- Structured logging is plain-text, not JSON; no OpenTelemetry/Datadog/LangSmith.
- Test suite (`agent/tests/`) is unit tests with fully mocked LLMs/retrievers — good for code-logic regressions, zero coverage for identification accuracy or persona-quality regressions.
- **No golden dataset, no LLM-as-judge, no accuracy regression tracking** — for an ID-accuracy product, this is the single biggest blind spot. Langfuse already has the primitives to close it (Datasets, Scores, Evaluators) — this doesn't need new infra, see Phase 3.

---

## Phased Roadmap

### Phase 1 — Critical (stop the bleeding: leaks, crashes, security)

- [x] **Per-session capability token** (branch `phase1/production-hardening`) — `backend/session_registry.py` (rewritten: SQLite-backed, atomic `create()` INSERT so "is this a brand-new thread_id" and "register it" are one DB op, no check-then-set race), `backend/routers/chat.py`, `backend/routers/sessions.py`. First `/chat` call for a new `thread_id` gets a secret back in `ChatResponse.session_secret`; every later call (continuation, `GET /history`, `DELETE`) must present it via `X-Session-Secret`, checked with `hmac.compare_digest`. Frontend (`useSessions.ts`, `client.ts`) stores and resends it. `GET /api/audio/{filename}` intentionally left ungated (unguessable temp filename, non-secret content). 9 integration tests in `backend/tests/test_session_auth.py` cover creation/rejection/cross-session-isolation/history/delete/retry-after-validation-failure.
  - Caught by an independent no-context review pass and fixed before merge: (1) `backend/main.py`'s CORS `allow_headers` didn't list `X-Session-Secret`, so the browser preflight would have silently blocked every follow-up request cross-origin (the default dev setup) — added. (2) `registry.create()` ran before input validation/graph invocation, so a first request that failed (bad image, graph error) orphaned the session — the client never got the secret, but the thread_id was already permanently consumed. Fixed by wrapping the turn in `try/except HTTPException` and `registry.evict()`-ing on any failure for a newly-created session, so a retry with the same thread_id is treated as a fresh first call.
- [x] **SqliteSaver checkpointer** — `agent/src/wildlens/graphs.py` (`build_graph` now takes an injectable `checkpointer`, defaults to `MemorySaver` for tests/local), wired up in `backend/main.py` (same SQLite file as the session-secret table, `SESSIONS_DB_PATH` env var). `DELETE /api/sessions/{id}` now calls `graph.checkpointer.delete_thread()` for real.
- [x] **Bounded `identification_history`** — `agent/src/wildlens/state.py` (`_bounded_identification_history` reducer, `MAX_IDENTIFICATION_HISTORY=15`).
- [x] **Tenacity retry** around LLM calls in `node_summarize_history` / `node_generate_guide_persona` — `agent/src/wildlens/nodes.py` (`_invoke_with_retry`, 3 attempts/exponential backoff). On exhausted retries: summarize degrades to a no-op (self-heals next call), persona degrades to a fixed apology script + surfaced `error_message` (never breaks the "final_script always set" contract). Covered by `test_summarize_history_llm_failure_retries_then_noop` / `test_persona_llm_failure_retries_then_returns_apology_script`.
- [x] **TTS runtime-failure fallthrough** — `agent/src/wildlens/tts.py` (`synthesise_audio`: edge-tts/gTTS calls now wrapped so a runtime failure falls through the chain instead of raising uncaught; both-failed still returns the `"NO_TTS_ENGINE_INSTALLED"` sentinel chat.py already handles gracefully). `agent/tests/test_tts.py`.
- [x] **Retry + escalated logging around RAG sub-retrievers** — `agent/src/wildlens/rag/ranking.py` (`_call_with_retry`, 1 retry per sub-retriever call; exhausted-retry failures escalated from `log.warning` to `log.error` — true paging/alerting is Phase 3 structured-observability scope, no such channel exists yet). `test_failing_retriever_retries_then_degrades_to_empty_not_raise` in `test_rag.py`.
- [x] **Supabase credential split** — `agent/src/wildlens/data/supabase_store.py` (`SupabaseStore(role=...)`, `SUPABASE_INGEST_KEY` vs `SUPABASE_RUNTIME_KEY`, falls back to `SUPABASE_KEY` if unset). Note: the doc's original "anon/read-only key at runtime" framing turned out slightly wrong — `ranking.py`'s enrichment write-back (`_write_enrichment`) means the runtime process legitimately needs `INSERT`/`UPDATE` on `documents`, not pure read-only; the split is least-privilege (species/documents read + documents write) rather than strictly read-only. RLS policy sketch in the class docstring; actual Supabase role/policy setup is dashboard config outside this repo.

### Phase 2 — Optimization (cost, latency, quality)

- [x] **Intent-gated retrieval** — `node_check_relevance` (merged `6d131ed`, PR #9): classifies text turns `on_topic`/`small_talk`/`off_topic` via cheap heuristics before an LLM fallback; skips retrieval for small talk; targets retrieval at the mentioned species instead of always the last-identified one.
  - Follow-up gaps found in a Fable review pass of the merged implementation (small, worth a follow-up commit — not a redesign):
    - [ ] Generic-noun alias false positives — `agent/src/wildlens/data/species_lookup.py:30,94-99` (e.g. "roller"/"monkey" aren't in `_GENERIC_HEAD_NOUNS`, so "roller coaster"/"monkey business" get misrouted to a specific species). One-line fix: extend the exclusion set.
    - [ ] Small-talk phrase match can swallow real questions with no keyword overlap — `nodes.py:357` + `_WILDLIFE_KEYWORDS` (`nodes.py:241-251`) (e.g. "Hey, how fast can it run?" has no listed keyword, gets routed to `small_talk`, retrieval skipped → risk of an ungrounded answer).
    - [ ] **(Priority)** LLM relevance fallback has zero conversation context — `nodes.py:290-311` (`_RELEVANCE_PROMPT` sends the message in isolation, no last-identified-species/history hint; a contextual pronoun follow-up like "can it swim?" can be misclassified `off_topic` and wrongly refused — the only path in this feature that produces a false refusal rather than a quality degradation).
    - [ ] Multi-animal messages ("lion vs elephant, who wins?") only target one species for retrieval — acceptable known limitation, document rather than fix unless it becomes a real complaint.
- [ ] Model tiering for summarization — introduce a second, cheaper text-model instance for `node_summarize_history`; set explicit `max_tokens` on both `llm_vision` and `llm_text`
- [ ] Semantic/response caching — query-normalized cache (`diskcache` preferred over Redis at this scale) in front of the BM25/Pinecone/Tavily fan-out
- [ ] Decompose the persona node — split fact-reconciliation from narration in `node_generate_guide_persona` (`nodes.py:601-704`), or at minimum move intro-vs-follow-up branching into prompt templates rather than Python if/else

### Phase 3 — Scale (multi-provider resilience, observability, eval)

- [ ] LLM provider failover — `.with_fallbacks()` for `llm_vision`/`llm_text` secondary providers (`backend/main.py`)
- [ ] Postgres checkpointer — swap `SqliteSaver` for `PostgresSaver` only once multi-instance/concurrent deployment is actually needed
- [ ] HITL gate for low-confidence IDs — `interrupt()` tied into the existing `route_after_analysis` confidence gate
- [ ] Golden-dataset eval via Langfuse Datasets — no new eval *platform* needed; Langfuse is already integrated and every trace already carries species/confidence/threat_level metadata (`observability.py:96-104`). Seed a Langfuse Dataset (`upsertDataset` / `upsertDatasetItem`) with image+expected-species pairs, write a small runner script (this part is unavoidable — a thin script, not a parallel harness) that replays them through the graph tagged to a dataset run, and score correctness + persona hallucination via `createScore` / `upsertEvaluator` (an evaluator can also run automatically on a sample of live traffic for ongoing online eval, not just offline against the golden set). Run the full dataset on a schedule *and* on-demand around prompt/model changes (see Verification below) — not per-commit. Closes the most consequential blind spot in the whole review.
- [ ] Structured observability — JSON logging (`structlog` or stdlib JSON formatter) for the plain-Python logs Langfuse doesn't see. No separate token-usage aggregation needed — Langfuse's `CallbackHandler` already captures per-generation token usage/cost natively; query it via the `queryMetrics` API instead of building a parallel layer.

---

## Verification per phase

- **Phase 1**: done — `pytest agent/tests/ backend/tests/` (110 passing: reducer/retry/TTS/RAG-retry fallback behavior in `agent/tests/`, capability-token creation/rejection/cross-session-isolation/history/delete in `backend/tests/test_session_auth.py`, no external API keys required for either).
- **Phase 2**: manual latency comparison (small-talk turn before/after intent gating — already landed) and a cache-hit-rate check on repeated queries once caching lands.
- **Phase 3**: run the golden-dataset Langfuse dataset run against a known image set and confirm scores are stable before/after a prompt or model change.
