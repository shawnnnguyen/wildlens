# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

```
safari-guide/
└── agent/                    # Python LangGraph agent (LLM orchestration)
    ├── src/safari_guide/      # Main package — installed via `pip install -e agent/`
    │   ├── state.py           # SafariGuideState + WildlifeIdentification schema
    │   ├── graphs.py          # Graph builder + make_turn_input()
    │   ├── nodes.py           # All 7 LangGraph nodes
    │   ├── rag.py             # Hybrid BM25 + Pinecone retriever
    │   ├── tts.py             # TTS synthesis (edge-tts → gTTS → sentinel)
    │   ├── __main__.py        # Interactive CLI demo (run_chat)
    │   └── data/              # Data ingestion pipeline
    │       ├── ingest.py      # CLI entry point
    │       ├── fetcher.py     # EOL / Wikipedia / API Ninjas / IUCN clients
    │       ├── supabase_store.py  # Supabase read/write adapter
    │       ├── lila.py        # LILA BC camera-trap image downloader
    │       ├── ultralytics_dl.py  # Ultralytics African Wildlife dataset
    │       └── species_list.json  # 48 Serengeti species definitions
    ├── supabase/schema.sql    # SQL DDL for species / documents / image_files tables
    ├── tests/                 # pytest suite (no API keys needed)
    └── requirements.txt
```

This project is currently scoped to the agent only: identifying African wildlife species from a photo and answering follow-up questions about the identified animal. A FastAPI backend and Expo mobile frontend existed in this repo's history but were removed to keep focus on the core AI agent; see git history if resurrecting them later.

## Setup

Copy `.env.example` to `.env` and fill in the required keys:

```
# Required — core LLMs
GOOGLE_API_KEY=...           # Gemini 1.5 Flash (vision/multimodal)
DEEPSEEK_API_KEY=...         # DeepSeek Chat (text generation via OpenAI-compatible API)

# Required — vector + document store
PINECONE_API_KEY=...         # Pinecone cloud vector store
PINECONE_INDEX_NAME=safari-guide
SUPABASE_URL=...             # https://<project-ref>.supabase.co
SUPABASE_KEY=...             # service role key (for ingestion) / anon key (read-only)

# Optional — data ingestion sources
IUCN_API_KEY=...             # IUCN Red List API (mock used if absent)
API_NINJAS_KEY=...           # API Ninjas animal facts (skipped if absent)

# Optional — observability
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

Pinecone/Supabase are optional for local dev — `rag.py` degrades gracefully to a BM25-only / mock-document retriever when they're unreachable or unset.

Install the agent package (editable):
```
pip install -e agent/
```

## Commands

Run the agent's interactive CLI demo (from `agent/`):
```
python -m safari_guide
```

Run all agent tests (from `agent/`, no API keys needed):
```
pytest tests/
```

Run a single test file:
```
pytest tests/test_nodes.py
pytest tests/test_rag.py
```

### Data ingestion (one-time setup)

Before first use, populate Supabase and Pinecone (from `agent/`):
```
# Text only: EOL + Wikipedia + API Ninjas + IUCN → Supabase + Pinecone
python -m safari_guide.data.ingest --text

# All image sources (EOL URLs + LILA BC + Ultralytics)
python -m safari_guide.data.ingest --images

# Everything
python -m safari_guide.data.ingest --all

# Dry run — show what would be done without writing
python -m safari_guide.data.ingest --all --dry-run
```

Run the Supabase schema once (in Supabase dashboard SQL editor or psql):
```
agent/supabase/schema.sql
```

## Architecture

This is a **LangGraph-based multi-turn conversational agent** — a "Digital Safari Tour Guide." A tourist provides a photo and/or text question; the system responds with a scripted commentary as "Baako" (a 20-year Serengeti guide persona), optionally with synthesised audio.

### Two LLMs, one graph

`build_graph(llm_vision, llm_text, retriever)` in `agent/src/safari_guide/graphs.py` compiles a single `StateGraph` with a `MemorySaver` checkpointer. The caller passes a `thread_id`; all prior session state is restored automatically between turns.

- `llm_vision` — Gemini 1.5 Flash; used only in `node_analyze_image` for multimodal structured output
- `llm_text` — DeepSeek Chat (via OpenAI-compatible API); used in `node_summarize_history` and `node_generate_guide_persona`

### Graph topology (single compiled graph)

```
START → route_entry
  ├─[image_path set] → analyze_image
  │     ├─[conf < 0.60] → unclear_photo_fallback → generate_guide_persona
  │     └─[conf ≥ 0.60] → safety_check → summarize_history → retrieve_information → generate_guide_persona
  └─[user_message set] → summarize_history → retrieve_information → generate_guide_persona
                                                                             │
                                                                       route_audio
                                                                  ├─[voice_requested] → generate_audio → END
                                                                  └─[text only] → END
```

`final_script` is **always written** before `END` — including the fallback path. `audio_file_path` is only written when `voice_requested=True`.

### Dependency injection via closures

Nodes are pure functions `(state, *deps) -> dict`. In `build_graph()`, deps are bound as closures:
```python
def _analyze(s):   return node_analyze_image(s, llm_vision)
def _persona(s):   return node_generate_guide_persona(s, llm_text)
```
This keeps nodes independently testable with mocked deps and avoids module-level globals.

### State reducers (critical)

`SafariGuideState` in `state.py` uses two non-default reducers:

- `chat_history: Annotated[list[BaseMessage], add_messages]` — appends messages and deduplicates by `id` (IDs assigned at merge time by LangGraph)
- `identification_history: Annotated[list[dict], operator.add]` — list concatenation; every animal identified this session accumulates here and is never overwritten

All other fields use last-write-wins. Nodes that don't need to update a field return `{}` — this is a confirmed safe LangGraph no-op.

### Caller contract

Always use `make_turn_input()` from `graphs.py` — it resets the per-turn output fields (`final_script`, `audio_file_path`, `retrieved_facts`, `error_message`) to empty strings so stale checkpointed values don't bleed into the next turn:

```python
app.invoke(
    make_turn_input(image_path="lion.jpg", voice_requested=True),
    config={"configurable": {"thread_id": "session-1"}},
)
```

### RAG: hybrid retriever (`rag.py`)

`init_rag()` returns an `_EnsembleRetriever` (custom RRF implementation) combining:
- **BM25** (`BM25Retriever`) — keyword matching; corpus rebuilt in-memory from Supabase `documents` table at startup; falls back to `_MOCK_DOCUMENTS` if Supabase is unreachable
- **Pinecone** (`PineconeVectorStore`) — cloud vector store; 384-dim `all-MiniLM-L6-v2` embeddings; falls back to `_NullRetriever` if `PINECONE_API_KEY` is absent or init fails

Fusion uses Reciprocal Rank Fusion (RRF, `rrf_k=60`) with equal weights (0.5/0.5). If Pinecone is unavailable the retriever degrades gracefully to BM25-only.

### TTS (`tts.py`)

`synthesise_audio(script)` tries engines in order:
1. `edge-tts` (Microsoft Neural, voice `en-US-GuyNeural`) — async, run in a worker thread via `asyncio.run()` to avoid event-loop conflicts in async web frameworks
2. `gTTS` — simpler fallback
3. Returns sentinel `"NO_TTS_ENGINE_INSTALLED"` if neither is available

The agent writes audio to the OS temp directory (`tempfile.mkstemp`, `safari_` prefix) — it has no concept of a web-servable path.

### Data layer (`data/`)

- `species_list.json` — canonical list of 48 Serengeti species with `common_name`, `scientific_name`, `threat_level`, and handcrafted `safety_notes`
- `fetcher.py` — HTTP clients for EOL (Encyclopedia of Life), Wikipedia, API Ninjas, and IUCN Red List
- `supabase_store.py` — read/write adapter: upserts species/documents/images on ingest; loads all documents for BM25 rebuild at startup
- `lila.py` — downloads LILA BC / Snapshot Serengeti camera-trap images to Supabase Storage
- `ultralytics_dl.py` — downloads Ultralytics African Wildlife Dataset (buffalo, elephant, rhino, zebra) to Supabase Storage
- `ingest.py` — CLI orchestrator; runs one-time population of Supabase + Pinecone from all text/image sources

### Long-range memory

`node_summarize_history` fires when `len(chat_history) > SUMMARY_THRESHOLD` (default 10). It compresses all but the most recent 6 messages into `conversation_summary` using a rolling LLM call. The persona node injects this summary plus the last 6 messages and a one-line digest of all animals seen this session (`identification_history`) into the LLM context.

## Production swap points

- Checkpointer: `MemorySaver()` → `SqliteSaver`/Postgres-backed saver (zero graph changes) — needed for durable, multi-instance-safe chat history
- RAG corpus: replace `_MOCK_DOCUMENTS` in `rag.py` or populate Supabase via `ingest.py`
- TTS engine: replace `synthesise_audio()` in `tts.py` (e.g. ElevenLabs) — zero node changes
- Observability: set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` in `.env` for Langfuse tracing
