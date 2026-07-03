from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .graphs import build_graph, make_turn_input
from .rag import init_rag

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("safari_guide")


def _build_langfuse_callback():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        return None


def run_chat() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set.")
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise EnvironmentError("DEEPSEEK_API_KEY is not set.")

    llm_vision = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.35)
    llm_text   = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.35,
    )

    print("Loading RAG retriever …")
    retriever = init_rag()
    app       = build_graph(llm_vision, llm_text, retriever)

    langfuse_cb = _build_langfuse_callback()
    config: dict = {"configurable": {"thread_id": "chat-session"}}
    if langfuse_cb:
        config["callbacks"] = [langfuse_cb]

    print("\n" + "═" * 60)
    print("  Baako — Digital Safari Tour Guide")
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
            result = app.invoke(
                make_turn_input(
                    image_path=image_path,
                    user_message=user_message,
                    voice_requested=voice_requested,
                ),
                config=config,
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

        print(f"\nBaako: {result.get('final_script', '').strip()}")

        audio = result.get("audio_file_path", "")
        if audio:
            print(f"\n  [Audio saved: {audio}]")

        print()


if __name__ == "__main__":
    run_chat()
