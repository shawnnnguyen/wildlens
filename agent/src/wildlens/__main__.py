from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .graphs import build_graph, make_turn_input
from .logging_config import configure_logging
from .observability import init_langfuse, invoke_with_tracing
from .rag import init_rag

load_dotenv()
configure_logging()
log = logging.getLogger("wildlens")


def run_chat() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set.")
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise EnvironmentError("DEEPSEEK_API_KEY is not set.")

    llm_vision = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash"), temperature=0.35,
    )
    llm_text   = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.35,
    )

    print("Loading RAG retriever …")
    retriever = init_rag()

    langfuse_cb = init_langfuse()
    app = build_graph(llm_vision, llm_text, retriever, tracing_enabled=bool(langfuse_cb))

    config: dict = {"configurable": {"thread_id": "chat-session"}}
    if langfuse_cb:
        config["callbacks"] = [langfuse_cb]
        config["metadata"] = {
            "langfuse_session_id": "chat-session",
            "langfuse_tags": ["cli"],
        }

    print("\n" + "═" * 60)
    print("  Kate — Digital Safari Tour Guide")
    print("═" * 60)
    print("  Type your question, or share an image path.")
    print("  Commands:")
    print("    image:<path>   — analyse a wildlife photo")
    print("    --voice        — append to any message for audio")
    print("    quit           — exit")
    print("═" * 60 + "\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        voice_requested = "--voice" in raw
        text            = raw.replace("--voice", "").strip()

        image_path   = ""
        user_message = ""

        if text.lower().startswith("image:"):
            image_path = text[6:].strip()
            if not os.path.exists(image_path):
                print(f"  [!] File not found: {image_path}\n")
                continue
        else:
            user_message = text

        try:
            result = invoke_with_tracing(
                app,
                make_turn_input(
                    image_path=image_path,
                    user_message=user_message,
                    voice_requested=voice_requested,
                ),
                config,
                langfuse_cb,
            )
        except Exception as exc:
            print(f"\n  [Error] {exc}\n")
            continue

        print()

        ident = result.get("identification_result") or {}
        if ident.get("species"):
            conf = ident.get("confidence_score", 0)
            print(f"  [{ident['species']} — {conf:.0%} confidence | threat: {ident.get('threat_level','?')}]")

        if result.get("error_message"):
            print(f"  [!] {result['error_message']}")

        print(f"\nKate: {result.get('final_script', '').strip()}")

        audio = result.get("audio_file_path", "")
        if audio:
            print(f"\n  [Audio saved: {audio}]")

        print()


if __name__ == "__main__":
    run_chat()
