# WildLens

**Snap a photo of wildlife and get an instant species ID, threat level, and a conversation with Kate, your AI safari guide, who can answer follow-up questions — grounded in a hybrid retrieval pipeline, not just a raw LLM guess.**

## Features

- **Vision-based species identification** — Gemini multimodal model returns species, genus, confidence score, visual traits, threat level, and habitat context from a single photo.
- **Kate, the conversational guide persona** — DeepSeek-powered chat that narrates findings and answers follow-up questions, with sliding-window history summarization to stay coherent over long sessions.
- **Hybrid RAG pipeline** — BM25 + Pinecone vector search + Tavily live web search, fused with reciprocal rank fusion, so answers are grounded in a real corpus instead of hallucinated.
- **Text-to-speech narration** — edge-tts (preferred) with a gTTS fallback, so responses can be read aloud.
- **Session security & persistence** — capability-token session auth (`X-Session-Secret`) and a SQLite-backed LangGraph checkpointer that survives restarts.
- **Built-in observability** — Langfuse tracing wired through every graph node, with image redaction on traces.

## Prerequisites

- **Python** 3.10+ (agent package), 3.11+ recommended
- **Node.js** 18+ and npm (frontend)
- **API keys** for Google AI Studio (Gemini) and DeepSeek at minimum — see [Configuration](#-configuration)
- At least one TTS engine installed (`edge-tts` or `gTTS`, both installable via pip)

## Installation

```bash
# Clone the repo
git clone <repository-url>
cd wildlens

# --- Agent (LangGraph orchestration package) ---
cd agent
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cd ..

# --- Backend (FastAPI service, imports the agent package above) ---
cd backend
pip install -r requirements.txt
cd ..

# --- Frontend (React + Vite) ---
cd frontend
npm install
cd ..
```

> The backend imports the `wildlens` package directly, so install the agent package (`pip install -e .` above) into the same virtual environment the backend runs in.

## Configuration

Copy `.env.example` to `.env` in the project root and fill in your keys:

```bash
# ── Required: Google Gemini (vision/multimodal) ────────────────────────────
GOOGLE_API_KEY=your_google_ai_studio_key_here
GEMINI_VISION_MODEL=gemini-2.5-flash

# ── Required: DeepSeek (text generation) ───────────────────────────────────
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# ── Backend session persistence (optional, has a default) ──────────────────
SESSIONS_DB_PATH=safari_sessions.db

# ── Pinecone (vector store) ─────────────────────────────────────────────────
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=safari-guide

# ── Supabase (document store + image storage) ───────────────────────────────
SUPABASE_URL=https://xxxxxxxxxxxxxxxxxxxx.supabase.co
SUPABASE_KEY=your_supabase_service_role_key_here
# SUPABASE_RUNTIME_KEY=your_supabase_least_privilege_key_here
# SUPABASE_INGEST_KEY=your_supabase_service_role_key_here

# ── Data sources for ingestion ──────────────────────────────────────────────
IUCN_API_KEY=your_iucn_api_key_here
API_NINJAS_KEY=your_api_ninjas_key_here

# ── Tavily (live web search — supplements RAG when the internal corpus is thin)
TAVILY_API_KEY=your_tavily_api_key_here
TAVILY_DAILY_CALL_CAP=500

# ── Langfuse observability (optional) ────────────────────────────────────────
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
```

For the frontend, optionally set the backend URL (defaults to `http://localhost:8000`):

```bash
# frontend/.env
VITE_API_BASE_URL=http://localhost:8000
```

## Usage

```bash
# Start the backend API (FastAPI + Uvicorn)
cd backend
uvicorn main:app --reload

# In a separate terminal, start the frontend dev server
cd frontend
npm run dev

# Run the agent as a standalone terminal chat (no backend/frontend needed)
cd agent
python -m wildlens

# Run tests
cd agent && pytest
cd backend && pytest
```
