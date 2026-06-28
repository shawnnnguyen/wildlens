# Digital Safari Tour Guide — Implementation Plan

> **Status:** Reviewed + UX requirements updated (2026-06-25). See §16 for architecture review findings.

---

## 1. Project Goal

Build a `src/`-packaged Python backend for a mobile "Digital Safari Tour Guide."  
A tourist starts a conversation — optionally with a photo — and can keep chatting about what they saw for as long as they like.  The system maintains full memory across every turn.

**Three hard UX requirements (updated 2026-06-25):**

1. **Single conversational thread** — there is no separate "primary" and "follow-up" mode. Every invocation goes through the same graph and the same `thread_id` session. Photo turns and text-only turns are handled by an entry router inside one graph.
2. **Persistent memory** — `MemorySaver` checkpointer + `thread_id` session key means the graph automatically restores all prior `chat_history` and `identification_result` on every turn. The caller never marshals history manually.
3. **Text is always returned; audio is always optional** — `final_script` (plain text) is populated on every path through the graph, including the unclear-photo fallback. TTS runs only when `voice_requested: bool` is `True` in the incoming state. The mobile client decides whether to request audio per turn.

Everything runs on **free / open-source tiers only** (portfolio project).

---

## 2. Technology Choices & Rationale

| Concern | Choice | Why |
|---|---|---|
| Orchestration | `langgraph >= 0.2` | StateGraph + conditional edges; built-in checkpointing for future persistence |
| Vision / LLM | `langchain-google-genai` → Gemini 1.5 Flash | Free tier; multimodal; function-calling for structured output |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace | Runs fully offline on CPU; 384-dim; fast enough for < 10k docs |
| Vector store | `FAISS` (CPU) | Zero infrastructure; file-level persistence; production-swappable to Chroma/Pinecone |
| TTS primary | `edge-tts` | Microsoft Neural voices; free; async; high quality for portfolio demos |
| TTS fallback | `gTTS` | Simpler; also free; requires internet |
| Observability | LangSmith | Zero-code wiring via env vars; traces every LLM + retrieval call |

---

## 3. File & Directory Layout

> Switched from single file to package after review (C2 / §15 Q1).

```
safari-guide/
├── src/
│   └── safari_guide/
│       ├── __init__.py
│       ├── __main__.py          ← entry point: python -m safari_guide
│       ├── state.py             ← SafariGuideState TypedDict + WildlifeIdentification Pydantic
│       ├── nodes.py             ← all six node functions (closure-param signature)
│       ├── graphs.py            ← build_graph() — one graph with MemorySaver checkpointer
│       ├── rag.py               ← init_rag(), mock corpus, FAISS persistence
│       └── tts.py               ← synthesise_audio(), edge-tts / gTTS logic
├── tests/
│   ├── test_nodes.py
│   └── test_rag.py
├── requirements.txt
├── .env.example
├── faiss_wildlife_index/        ← auto-created on first run
├── audio_output/                ← auto-created; generated .mp3 files
└── IMPLEMENTATION_PLAN.md
```

---

## 4. Graph State Schema

```python
class SafariGuideState(TypedDict):
    # ── Inputs (caller sets these on each invoke) ──────────────────────────
    image_path:             str    # path or base64 URI; empty string on text-only turns
    user_message:           str    # tourist's text; empty string on photo-only turns
    voice_requested:        bool   # True → run TTS node; False → skip it

    # ── Conversation memory (managed by checkpointer) ─────────────────────
    chat_history:           Annotated[list[BaseMessage], add_messages]

    # ── Long-range memory (updated 2026-06-25) ────────────────────────────
    identification_history: Annotated[list[dict], operator.add]
    # Accumulates every animal identified this session via operator.add reducer.
    # Never overwritten — each new photo appends a new dict.
    # Enables cross-animal questions: "compare this to the lion we saw earlier."

    conversation_summary:   str
    # LLM-generated compression of older chat_history turns.
    # Injected before the 6-message sliding window so Baako can answer
    # questions about animals discussed many turns ago without a growing context window.
    # Updated by node_summarize_history when len(chat_history) > SUMMARY_THRESHOLD (10).

    # ── Pipeline outputs (written by nodes; read by downstream nodes) ─────
    identification_result:  dict   # most recent animal only — see sub-keys below
    retrieved_facts:        str
    final_script:           str    # ALWAYS populated — every path sets this
    audio_file_path:        str    # set only when voice_requested=True

    # ── Routing signal ────────────────────────────────────────────────────
    error_message:          str    # set on fallback or exception; empty otherwise
```

**`voice_requested` is the audio gate.**  Every node always writes `final_script`.  The `generate_audio` node only runs when `voice_requested is True`.  This means the caller receives a text answer on every invocation and can optionally request speech by flipping one flag — no second API call needed.

**Caller contract:**

```python
# Turn 1 — new photo, want audio
app.invoke(
    {"image_path": "lion.jpg", "user_message": "", "voice_requested": True},
    config={"configurable": {"thread_id": "tour-session-1"}},
)

# Turn 2 — follow-up question, text only
app.invoke(
    {"image_path": "", "user_message": "How fast can it run?", "voice_requested": False},
    config={"configurable": {"thread_id": "tour-session-1"}},
)

# Turn 3 — another question, wants audio this time
app.invoke(
    {"image_path": "", "user_message": "Is it endangered?", "voice_requested": True},
    config={"configurable": {"thread_id": "tour-session-1"}},
)
```

The checkpointer restores `chat_history` and `identification_result` automatically between turns — the caller only ever passes the *new* inputs for that turn.

**Key reducer decision:** `chat_history` uses `add_messages` (from `langgraph.graph.message`) not `operator.add`.  The reducer assigns UUIDs at merge time and deduplicates by `id` — required for safe checkpointing and streaming.  All other fields use the implicit last-write-wins reducer.

**`identification_result` sub-keys** (populated by `node_analyze_image`, mutated by `node_safety_check`):

| Key | Type | Set by |
|---|---|---|
| `species` | str | `node_analyze_image` |
| `confidence_score` | float 0–1 | `node_analyze_image` |
| `visual_traits` | list[str] | `node_analyze_image` |
| `threat_level` | "low" \| "medium" \| "high" | `node_analyze_image` |
| `habitat_context` | str | `node_analyze_image` |
| `safety_warning` | str | `node_safety_check` (injected only if `threat_level == "high"`) |

---

## 5. Structured Output Schema (Pydantic)

```python
from typing import Literal
from pydantic import BaseModel, Field

class WildlifeIdentification(BaseModel):
    species:          str
    confidence_score: float = Field(ge=0.0, le=1.0)   # bounds enforced by Pydantic
    visual_traits:    list[str]
    threat_level:     Literal["low", "medium", "high"] # Literal prevents "Low"/"High!" silently bypassing == "high" checks
    habitat_context:  str
```

> D2 fix: `threat_level` changed from `str` to `Literal["low","medium","high"]` and `confidence_score` bounds moved to `Field` so they are enforced at validation, not just documented.

Used with `llm.with_structured_output(WildlifeIdentification)`.  
Gemini function-calling enforces the schema; downstream nodes receive a plain `dict` via `.model_dump()`.

---

## 6. Node Inventory

### 6.1 `node_analyze_image`
- **Input state fields:** `image_path`
- **Behaviour:** Validates file extension + size; encodes to base64 data URI; sends multimodal `HumanMessage`; uses `with_structured_output(WildlifeIdentification)` to enforce schema. Wrapped in try/except — on failure sets `error_message` and `confidence_score=0.0` so the graph routes to the fallback.
- **Output keys:** `identification_result` (overwritten — most recent animal), `identification_history` (appended via `operator.add`), `chat_history` (two seed messages), `error_message` (on failure)

### 6.2 `node_unclear_photo_fallback`
- **Triggered when:** `confidence_score < MIN_CONFIDENCE`
- **Behaviour:** Composes a polite "please retry" message; does **not** call the LLM (zero token cost on this path). Sets `final_script` — **text is always returned**.
- **Output keys:** `final_script`, `error_message`, `chat_history`

### 6.3 `node_safety_check`
- **Behaviour:** If `threat_level == "high"`, injects a `safety_warning` key into a **new copy** of `identification_result` (pure function — never mutates in-place). Returns `{}` otherwise (LangGraph no-op, guaranteed stable).
- **Output keys:** `identification_result` (updated copy) or `{}` (no-op)

### 6.4 `node_summarize_history`  *(new — 2026-06-25)*
- **Triggered when:** `len(chat_history) > SUMMARY_THRESHOLD` (default 10 messages)
- **Behaviour:** Summarizes all messages except the most recent 6 into a 3–5 sentence `conversation_summary` string using one LLM call. On subsequent calls it incorporates the prior summary as context, so it is always a rolling compression rather than a full replay. Returns `{}` when below threshold (guaranteed no-op).
- **Placement in graph:** Runs before `retrieve_information` on both the photo path (after `safety_check`) and the text-only path (directly after `route_entry`).
- **Output keys:** `conversation_summary` or `{}` (no-op)

### 6.5 `node_retrieve_information`
- **Input state fields:** `identification_result.species`, `user_message`
- **Query strategy:** `f"{species} {user_message}".strip()` — context-aware for both photo turns and text-only follow-ups in one retriever.
- **Output keys:** `retrieved_facts`

### 6.6 `node_generate_guide_persona`
- **Always writes `final_script`** — this is the guarantee that every turn returns text.
- **Handles two modes** via `user_message` presence:
  - *Photo turn* — full introduction with optional safety alert prefix and guidebook facts.
  - *Text turn* — answers the tourist's question; incorporates prior conversation for multi-turn context.
- **LLM context stack (in order):**
  1. Baako system prompt
  2. `conversation_summary` block (if non-empty) — compressed long-range memory
  3. Last 6 `chat_history` messages — recent turn context
  4. `identification_history` digest — one-line per animal seen this session, for cross-animal questions
  5. Current task message
- **Persona:** "Baako" — 20-year Serengeti guide; conversational, punchy, audio-optimised (no markdown, 140–220 words).
- **Output keys:** `final_script`, `chat_history` (this turn appended via `add_messages`)

### 6.7 `node_generate_audio`  *(conditional — only runs when `voice_requested=True`)*
- **Behaviour:** Reads `final_script`; delegates to `synthesise_audio()` utility. Thin adapter — TTS engine swap (e.g., ElevenLabs) requires zero graph changes.
- **Output keys:** `audio_file_path`
- **When skipped:** `audio_file_path` remains `""` (its checkpointed value from prior turn, or empty string on first turn). Caller checks `audio_file_path != ""` to know whether audio is available.

---

## 7. Control Flow

> One graph, `MemorySaver` checkpointer, `thread_id` session key. Audio is a per-turn flag, not a separate path.

### 7.1 Routing functions

| Function | Reads | Returns |
|---|---|---|
| `route_entry` | `image_path` (non-empty = new photo) | `"analyze_image"` or `"retrieve_information"` |
| `route_after_analysis` | `identification_result.get("confidence_score", 0.0)` | `"unclear_photo_fallback"` or `"safety_check"` |
| `route_audio` | `voice_requested` | `"generate_audio"` or `END` |

### 7.2 Full graph (single compiled graph)

```
START
  └─► route_entry
        │
        ├─[image_path non-empty]──► analyze_image
        │                               │
        │               ┌───────────────┴──────────────────┐
        │               │ conf < MIN_CONFIDENCE            │ conf ≥ MIN_CONFIDENCE
        │               ▼                                  ▼
        │    unclear_photo_fallback               safety_check
        │               │                                  │
        │               │                        retrieve_information
        │               │                                  │
        │               └──────────────────► generate_guide_persona
        │                                              │
        │                                        route_audio
        │                                     ┌──────┴──────┐
        │                              voice_requested    text only
        │                                     ▼               ▼
        │                              generate_audio        END
        │                                     │
        │                                    END
        │
        └─[user_message non-empty]──► retrieve_information
                                               │
                                    generate_guide_persona
                                               │
                                         route_audio
                                      ┌──────┴──────┐
                               voice_requested    text only
                                      ▼               ▼
                               generate_audio        END
                                      │
                                     END
```

**Key design points:**

- `final_script` is set on **every path**, including `unclear_photo_fallback` — text is never missing.
- `generate_audio` is a **leaf node** only reached when `voice_requested=True`. It reads `final_script` and writes `audio_file_path`.
- The checkpointer restores all state between turns automatically. The caller only passes new inputs per turn (`image_path`, `user_message`, `voice_requested`).

### 7.3 Checkpointer setup

```python
from langgraph.checkpoint.memory import MemorySaver

app = build_graph(llm, vectorstore).compile(checkpointer=MemorySaver())

# Every turn uses the same thread_id — history accumulates automatically
config = {"configurable": {"thread_id": "tour-session-42"}}

app.invoke({"image_path": "lion.jpg", "user_message": "", "voice_requested": True},  config=config)
app.invoke({"image_path": "",         "user_message": "How fast?",  "voice_requested": False}, config=config)
app.invoke({"image_path": "",         "user_message": "Endangered?","voice_requested": True},  config=config)
```

Swap `MemorySaver()` → `SqliteSaver.from_conn_string("safari.db")` for persistent production sessions; zero graph changes required.

---

## 8. RAG Initialisation Strategy

- **First run:** Build FAISS index from 10 mock wildlife guidebook `Document` objects; persist to `faiss_wildlife_index/` on disk.
- **Subsequent runs:** Reload from disk (fast path via `FAISS.load_local`).
- **Mock corpus includes:** African Lion, African Elephant, Reticulated Giraffe, African Leopard, Plains Zebra, Hippopotamus, African Wild Dog, Cheetah, Nile Crocodile, Umbrella Thorn Acacia.
- **Production migration path:** Replace `Document` list with `PyMuPDFLoader` + `RecursiveCharacterTextSplitter`; re-index; same graph code unchanged.

---

## 9. Audio Generation Strategy

```
synthesise_audio(script)
    │
    ├─[edge-tts available]──► _edge_tts_worker (async, Microsoft Neural "en-US-GuyNeural")
    │                           └─► _run_coroutine_in_thread()
    │                                  (dedicated thread pool loop — safe inside FastAPI / Jupyter)
    │
    ├─[gTTS available]──► gTTS(text, lang="en").save(path)
    │
    └─[neither]──► log warning; return "NO_TTS_ENGINE_INSTALLED"
```

`edge-tts` is async; called via a **worker thread with its own event loop** to avoid `RuntimeError: This event loop is already running` when the graph is eventually embedded in an async web framework.

---

## 10. Dependency Injection Pattern

Nodes that require external resources (`llm`, `vectorstore`) receive them via **closures** defined inside the graph builder functions:

```python
def build_primary_graph(llm, vectorstore):
    def _analyze(s):  return node_analyze_image(s, llm)
    def _retrieve(s): return node_retrieve_information(s, vectorstore)
    ...
    graph.add_node("analyze_image", _analyze)
```

**Why not globals?**  
Closures make each node a pure function, enable parallel test instantiation with mocked dependencies, and produce no hidden coupling between graph construction and module-level state.

---

## 11. State Input Helpers

Because the checkpointer restores full state between turns, the caller only ever needs to pass the **delta** for the current turn. One helper covers all cases:

```python
def make_turn_input(
    image_path: str = "",
    user_message: str = "",
    voice_requested: bool = False,
) -> dict:
    """
    Build the minimal input dict for one graph invocation.
    The checkpointer merges this with restored state from the prior turn.
    """
    return {
        "image_path":      image_path,
        "user_message":    user_message,
        "voice_requested": voice_requested,
    }
```

There is no `make_followup_state` — that manual state-threading pattern is eliminated by the checkpointer. The `SafariGuideState` TypedDict is still the full schema (nodes read and write all fields); only the *caller-facing input* is simplified to three fields per turn.

---

## 12. Demo Invocation Block

A single `run_demo()` function drives one unified conversational session using `thread_id="demo-session-1"`.  No state is marshalled between turns — the checkpointer handles it.

```python
config = {"configurable": {"thread_id": "demo-session-1"}}

# Turn 1 — Tourist snaps a photo; wants audio narration
r1 = app.invoke(
    {"image_path": "sample_lion.jpg", "user_message": "", "voice_requested": True},
    config=config,
)
# → r1["final_script"]    always present (text)
# → r1["audio_file_path"] present (voice_requested=True)

# Turn 2 — Follow-up question; text only
r2 = app.invoke(
    {"image_path": "", "user_message": "How fast can it run?", "voice_requested": False},
    config=config,
)
# → r2["final_script"]    always present
# → r2["audio_file_path"] == "" (skipped)

# Turn 3 — Another question; tourist now wants audio
r3 = app.invoke(
    {"image_path": "", "user_message": "Is it endangered?", "voice_requested": True},
    config=config,
)
# → r3["final_script"]    always present
# → r3["audio_file_path"] present

# Turn 4 — New animal entirely (new photo, same session — history preserved)
r4 = app.invoke(
    {"image_path": "sample_elephant.jpg", "user_message": "", "voice_requested": False},
    config=config,
)
```

**What the caller always receives:**

| Field | Guarantee |
|---|---|
| `final_script` | Always a non-empty string — on every turn, every path |
| `audio_file_path` | Non-empty string path when `voice_requested=True`; `""` otherwise |
| `error_message` | Non-empty if something went wrong (low confidence, API failure) |
| `chat_history` | Full accumulated conversation (managed by checkpointer) |

---

## 13. `requirements.txt` Content

> Version matrix corrected after review (A2): all packages pinned to the **`langchain-core 0.3.x` era**.

```
# Orchestration
langgraph>=0.2.0

# LangChain — all pinned to core 0.3.x era (must be internally consistent)
langchain-core>=0.3.0,<0.4
langchain-google-genai>=2.0.0        # 2.x targets core 0.3; 1.x targets core 0.2
langchain-community>=0.3.0           # 0.3.x targets core 0.3; carries FAISS
langchain-huggingface>=0.1.0         # 0.1.x targets core 0.3

# Vector store
faiss-cpu>=1.7.4

# Embeddings
sentence-transformers>=2.2.0

# Structured output
pydantic>=2.0.0

# TTS
edge-tts>=6.1.9
# gTTS>=2.3.2      # uncomment as fallback if edge-tts unavailable

# Env management
python-dotenv>=1.0.0
```

---

## 14. `.env.example` Content

```
# Required
GOOGLE_API_KEY=your_google_ai_studio_key_here

# Optional — LangSmith observability
# Both name pairs are accepted; LANGSMITH_* is the newer canonical form.
# LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY still work but are being superseded.
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=safari-guide

# Optional — tunable constants (defaults shown)
MIN_CONFIDENCE=0.60
```

---

## 15. Open Questions — Resolved

> All six questions were reviewed by an independent Opus agent. Decisions are recorded here.

1. **Single file vs. package → Package wins.**  
   Go with `src/safari_guide/` (`state.py`, `nodes.py`, `graphs.py`, `rag.py`, `tts.py`, `__main__.py`). This better demonstrates software engineering discipline and makes the DI pattern testable in isolation.

2. **`add_messages` ID safety → Safe, for the right reason.**  
   The reducer itself assigns UUIDs to messages lacking an `id` at merge time — not the constructor. This makes deduplication safe regardless of `langchain-core` version, as long as the `add_messages` import is from `langgraph.graph.message`. No code change needed; rationale comment updated.

3. **Confidence threshold → Make it a named constant, env-overridable.**  
   `MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.60"))` at module level. Note: LLM self-reported confidence is poorly calibrated; treat this gate as UX hygiene, not a precision signal.

4. **Two graphs vs. one + checkpointer → One graph + `MemorySaver` checkpointer.**  
   The two-graph split was motivated by fear of re-running image analysis on follow-ups — solved trivially with a conditional entry edge. One graph with `MemorySaver` (demo) / `SqliteSaver` (prod) and `thread_id` routing is the idiomatic LangGraph conversational pattern and aligns with the stated §2 checkpointing rationale.

5. **`node_safety_check` returning `{}` → Guaranteed safe.**  
   Confirmed stable across LangGraph 0.2.x and 0.3.x. Each channel keeps its current value when no update arrives. Keep the pattern.

6. **TTS async thread-pool → Keep the pattern, simplify the worker.**  
   Dedicated worker thread + `asyncio.run()` (not manual `new_event_loop`/`set_event_loop`) is the correct approach. Avoids the "already running" error in FastAPI/Jupyter. `asyncio.run()` is safe inside a freshly spawned thread because new threads have no event loop.

---

## 16. Architecture Review Findings

> Conducted by an independent Opus agent with zero context from the planning conversation.  
> Items are ordered by severity: **[A] = will crash, [B] = LangGraph correctness, [C] = architecture, [D] = RAG, [E] = TTS, [F] = missed concerns**.

### [A] Hard crashes / runtime errors to fix before coding

| # | Finding | Fix |
|---|---|---|
| A1 | `FAISS.load_local(path, embeddings)` raises `ValueError` since `langchain-community ~0.2` — must pass `allow_dangerous_deserialization=True` | Add the flag to every `load_local` call |
| A2 | Dependency matrix is internally inconsistent — `langchain-core>=0.3` requires `langchain-google-genai>=2.0` and `langchain-community>=0.3`; the plan pins 1.x / 0.2.x versions | Updated §13 pinned to the core-0.3 matrix |
| A3 | `route_after_analysis` reads `identification_result["confidence_score"]` directly — KeyErrors if structured output returns `{}` on API/parse failure | Use `.get("confidence_score", 0.0)` — failed parse routes to fallback instead of crashing |

### [B] LangGraph correctness

- **B1** — Empty-dict `{}` no-op (§6.3) is confirmed stable. Keep it.
- **B2** — `add_messages` assigns IDs at merge time, not at construction. Plan rationale corrected in §15 Q2.
- **B3** — Sequential flow means no concurrent write conflicts on `identification_result`. Safe as-is.
- **B4** — §2 cites checkpointing as a LangGraph rationale but neither graph uses a checkpointer. Resolved by the Q4 decision: one graph + `MemorySaver`.

### [C] Architecture decisions

- **C1** — Switch from two compiled graphs to **one graph + `MemorySaver` checkpointer**. See §15 Q4.
- **C2** — Switch from single file to **`src/safari_guide/` package**. See §15 Q1.
- **C3** — Closures for DI (§10) confirmed correct. No changes.

### [D] RAG design

- **D1** — LLM self-reported confidence is uncalibrated. Keep the gate as a UX fallback, not a precision signal.
- **D2** — `threat_level: str` in Pydantic must be `Literal["low", "medium", "high"]`. Otherwise Gemini can return `"Low"` and the `== "high"` check in `node_safety_check` silently fails.
- **D3** — Specify `k=3` explicitly in retriever `search_kwargs` and document how `retrieved_facts` is assembled from `doc.page_content`.
- **D4** — FAISS-CPU + all-MiniLM-L6-v2 confirmed right choice. `pymupdf` noted as future dep.

### [E] TTS

- **E1** — Thread-pool + own event loop pattern is correct.
- **E2** — Simplify the worker to `asyncio.run(coro)` inside the thread — no manual loop management needed.
- **E3** — Use UUID-based audio filenames (already planned) to prevent clobber under concurrent API requests.

### [F] Gaps to address in implementation

| # | Gap | Resolution |
|---|---|---|
| F1 | No error-handling nodes — exceptions in `node_analyze_image` kill the graph; `error_message` is never set by anything except the fallback node | Add try/except in nodes; set `error_message` on failure; add `node_error → END` terminal edge |
| F2 | Unclear-photo fallback path ends without TTS — mobile client gets no audio | Route `unclear_photo_fallback` through `generate_audio` before `END` |
| F3 | `image_path` is unvalidated — arbitrary file read + memory-exhaustion if exposed via API | Add extension/MIME check and file-size cap in `node_analyze_image` preamble |
| F4 | `make_initial_state` must initialise containers to `{}` and `[]`, not empty strings | Explicitly initialise `identification_result={}`, `chat_history=[]` |
| F5 | `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` are being superseded by `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | Support both in `.env.example`; add a comment noting the alias |
| F6 | No env-var presence check at startup — missing `GOOGLE_API_KEY` surfaces as an opaque mid-graph error | Validate at boot in `run_demo()` / graph builder |
| F7 | No test strategy — undercuts the "production-patterned" claim | Add unit tests per node with mocked `llm` and in-memory FAISS (reinforced by package layout) |

---

*(Moved to §15 with resolved decisions after the architecture review.)*
