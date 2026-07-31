"""
Replays the "wildlens-golden-v1" Langfuse dataset (see seed_dataset.py)
through the compiled graph and scores each item's identification accuracy,
threat-level accuracy, confidence calibration, and persona-script factual
grounding.

Not run per-commit: invoked manually (see --limit for a quick smoke run) or
on a schedule / around prompt-and-model changes via the GitHub Actions
workflow at .github/workflows/eval.yml.

Online evaluators (scoring a sample of live production traffic, as opposed
to this offline dataset replay) have no Python SDK surface as of
langfuse==4.12 — configure those in the Langfuse UI (Evaluators tab)
directly rather than here.

Usage:
    python -m wildlens.eval.run_eval [--limit N] [--concurrency N] [--run-name NAME]

Env vars:
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY  — required
    GOOGLE_API_KEY                             — vision + hallucination judge
    DEEPSEEK_API_KEY                           — persona generation
    SUPABASE_URL                               — public image fetch (no key needed)
    PINECONE_API_KEY / SUPABASE_RUNTIME_KEY / TAVILY_API_KEY — optional; RAG
        degrades gracefully without them (see rag/factory.py)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

from ..data.supabase_store import public_image_url
from ..graphs import build_graph, make_turn_input
from ..logging_config import configure_logging
from ..observability import init_langfuse
from ..rag import init_rag
from .scorers import (
    EVAL_TASK_ERROR,
    confidence_calibration,
    make_persona_hallucination_evaluator,
    mean_confidence_when_wrong,
    species_match,
    threat_level_match,
)

log = logging.getLogger("wildlens.eval.run_eval")

DATASET_NAME = "wildlens-golden-v1"


def _load_species_ground_truth() -> dict[str, dict]:
    path = Path(__file__).parent.parent / "data" / "species_list.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {entry["common_name"]: entry for entry in entries}


def _download_image(storage_path: str) -> str:
    """
    Fetch a dataset item's image from the public wildlife-images bucket to a
    temp file. make_turn_input's image_path expects a local path or a
    data:… URI (see graphs.py/nodes.py's _to_data_uri) — not raw bytes or a
    remote URL — so this download step is unavoidable per item.
    """
    url = public_image_url(storage_path)
    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    suffix = Path(storage_path).suffix or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)
    except BaseException:
        # mkstemp already created the file on disk before the write — don't
        # leak it if the write itself fails (disk full, interrupted).
        os.unlink(tmp_path)
        raise
    return tmp_path


def make_task(graph, langfuse_handler):
    """
    Closure binding the shared compiled graph + Langfuse callback handler —
    same dependency-injection-via-closure pattern as graphs.py's node
    bindings. Built once outside run_experiment, not per item: re-running
    init_rag()/build_graph() per item would make even a small run take
    forever (init_rag loads a HuggingFace embedding model and connects to
    Pinecone — see rag/factory.py).

    Item isolation comes from a unique thread_id per item on this one shared
    graph/checkpointer, exactly how the backend isolates concurrent real
    sessions (see backend/routers/chat.py) — not from a fresh graph per item.
    """

    def task(*, item, **_kwargs) -> dict:
        tmp_path: str | None = None
        try:
            tmp_path = _download_image(item.input)
            thread_id = f"eval-{item.id}"
            turn_input = make_turn_input(image_path=tmp_path, user_message="", voice_requested=False)
            config: dict = {"configurable": {"thread_id": thread_id}}
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]
                config["metadata"] = {"langfuse_session_id": thread_id, "langfuse_tags": ["eval"]}
            result = graph.invoke(turn_input, config)
        except Exception as exc:
            # A download or graph-invocation failure here is an eval-harness
            # infra problem (network blip, transient API error), not a model
            # judgment. Returning an explicit error-shaped output — instead
            # of letting the exception propagate — keeps the item in the run
            # and scored distinctly as "error" (see scorers.py's
            # _is_eval_task_error), rather than silently vanishing from
            # item_results: the SDK drops any item whose task() raises, with
            # no visible signal in the run's aggregate scores (verified
            # against langfuse's run_experiment — a task exception is caught,
            # logged, and the item excluded entirely).
            log.warning("Eval item %s failed: %s", item.id, exc)
            return {
                "final_script": "",
                "identification_result": {},
                "current_analysis": {},
                "error_message": f"{EVAL_TASK_ERROR}: {exc}",
            }
        finally:
            if tmp_path is not None:
                os.unlink(tmp_path)

        # Shape scorers.py's evaluators expect — see its module docstring.
        return {
            "final_script": result.get("final_script", ""),
            "identification_result": result.get("identification_result") or {},
            "current_analysis": result.get("current_analysis") or {},
            "error_message": result.get("error_message") or "",
        }

    return task


def run(limit: int | None = None, concurrency: int = 3, run_name: str | None = None) -> None:
    from langfuse import get_client

    langfuse_handler = init_langfuse()
    if langfuse_handler is None:
        raise EnvironmentError(
            "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY must be set to run the eval."
        )
    client = get_client()

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI

    vision_model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    llm_vision = ChatGoogleGenerativeAI(model=vision_model, temperature=0.35)
    llm_text = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0.35,
    )
    # Judging DeepSeek's persona output with DeepSeek itself is self-grading —
    # use Gemini (already a dependency, for vision) as an independent judge.
    judge_llm = ChatGoogleGenerativeAI(model=vision_model, temperature=0.0)

    log.info("Loading RAG retriever …")
    retriever = init_rag()
    graph = build_graph(llm_vision, llm_text, retriever, tracing_enabled=True)

    dataset = client.get_dataset(DATASET_NAME)
    items = dataset.items[:limit] if limit else dataset.items
    if not items:
        raise ValueError(
            f"Dataset '{DATASET_NAME}' has no items — run seed_dataset.py first."
        )
    log.info("Running eval on %d/%d items", len(items), len(dataset.items))

    species_ground_truth = _load_species_ground_truth()
    persona_hallucination = make_persona_hallucination_evaluator(judge_llm, species_ground_truth)

    result = client.run_experiment(
        name="wildlens-golden-eval",
        run_name=run_name,
        data=items,
        task=make_task(graph, langfuse_handler),
        evaluators=[species_match, threat_level_match, confidence_calibration, persona_hallucination],
        run_evaluators=[mean_confidence_when_wrong],
        # SDK default is 50 — left unset, a run fires 50 parallel Gemini-vision
        # + DeepSeek-persona turns and hits rate limits immediately.
        max_concurrency=concurrency,
        metadata={"vision_model": vision_model, "text_model": "deepseek-chat"},
        _dataset_version=dataset.version,
    )
    client.flush()
    print(result.format())


def main() -> None:
    load_dotenv()
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the golden-dataset accuracy eval against Langfuse.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N dataset items (smoke run).")
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="max_concurrency for run_experiment (default 3 — the SDK default of 50 will hit LLM rate limits).",
    )
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    run(limit=args.limit, concurrency=args.concurrency, run_name=args.run_name)


if __name__ == "__main__":
    main()
