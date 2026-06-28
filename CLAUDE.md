# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Copy `.env.example` to `.env` and fill in both required keys:
```
GOOGLE_API_KEY=...       # Gemini 1.5 Flash (vision/multimodal)
DEEPSEEK_API_KEY=...     # DeepSeek Chat (text generation via OpenAI-compatible API)
```

Install dependencies:
```
pip install -r requirements.txt
```

## Commands

Run the demo (5-turn conversational session):
```
python -m safari_guide
```

Run all tests (no API keys needed):
```
pytest tests/
```

Run a single test file:
```
pytest tests/test_nodes.py
pytest tests/test_rag.py
```

## Architecture

This is a **LangGraph-based multi-turn conversational agent** — a mobile "Digital Safari Tour Guide." A tourist sends a photo and/or text question; the system responds with a scripted commentary as "Baako" (a 20-year Serengeti guide persona), optionally with synthesised audio.

### Two LLMs, one graph

`build_graph(llm_vision, llm_text, retriever)` in `graphs.py` compiles a single `StateGraph` with a `MemorySaver` checkpointer. The caller passes a `thread_id`; all prior session state is restored automatically between turns.

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

`init_rag()` returns an `EnsembleRetriever` combining:
- **BM25** (`BM25Retriever`) — keyword matching; always rebuilt in-memory from the mock corpus
- **FAISS** (`langchain-community`) — semantic similarity via `all-MiniLM-L6-v2` (CPU, 384-dim); persisted to `faiss_wildlife_index/` on first run, reloaded on subsequent runs

Fusion uses Reciprocal Rank Fusion (RRF) with equal weights (0.5/0.5), returning `k=3` docs per retriever before fusion. FAISS requires `allow_dangerous_deserialization=True` on `load_local`.

### TTS (`tts.py`)

`synthesise_audio(script)` tries engines in order:
1. `edge-tts` (Microsoft Neural, voice `en-US-GuyNeural`) — async, run in a worker thread via `asyncio.run()` to avoid event-loop conflicts in async web frameworks
2. `gTTS` — simpler fallback
3. Returns sentinel `"NO_TTS_ENGINE_INSTALLED"` if neither is available

Audio files are UUID-named MP3s written to `audio_output/` (auto-created).

### Long-range memory

`node_summarize_history` fires when `len(chat_history) > SUMMARY_THRESHOLD` (default 10). It compresses all but the most recent 6 messages into `conversation_summary` using a rolling LLM call. The persona node injects this summary plus the last 6 messages and a one-line digest of all animals seen this session (`identification_history`) into the LLM context.

### Auto-created directories

- `faiss_wildlife_index/` — FAISS persistence (first run builds, subsequent runs reload)
- `audio_output/` — generated MP3 files

### Production swap points

- Checkpointer: `MemorySaver()` → `SqliteSaver.from_conn_string("safari.db")` (zero graph changes)
- RAG corpus: replace `_MOCK_DOCUMENTS` in `rag.py` with real loaders (e.g. `PyMuPDFLoader`)
- TTS engine: replace `synthesise_audio()` in `tts.py` (e.g. ElevenLabs) — zero node changes
- Observability: set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` in `.env` for Langfuse tracing
