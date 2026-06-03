"""UI GreenMetric RAG Assistant v1.0 — Gradio web interface."""

import os
import queue
import threading
import gradio as gr
from openai import APIError
from src.pipeline import ask, _budget

# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "DEEPSEEK_API_KEY not found. "
        "Set it in your .env file or as an environment variable."
    )


# ---------------------------------------------------------------------------
# ChromaDB startup check — auto-build if missing (HF Spaces deployment)
# ---------------------------------------------------------------------------


def _ensure_chromadb() -> None:
    """Build ChromaDB collection if it doesn't exist on disk.

    Handles the case where ``chroma_db/`` was not properly deployed to
    HF Spaces, or the collection was accidentally dropped.
    """
    import chromadb

    collection_name = os.getenv("RAG_COLLECTION", "greenmetric_bgem3")
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.get_collection(collection_name)
        print(f"ChromaDB collection '{collection_name}' found.")
    except ValueError:
        print(f"ChromaDB collection '{collection_name}' not found — building...")
        from build_collection import chunk_all
        from src.embedder import store

        sources = chunk_all()
        store(sources, collection_name=collection_name)
        print("ChromaDB build complete.")


_ensure_chromadb()


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------

def _budget_display() -> str:
    used = _budget.used()
    cap = _budget.daily_cap
    pct = min(100, used / cap * 100) if cap else 0
    return (
        f"{used:,} / {cap:,} tokens used today ({pct:.0f}%)\n\n"
        "*Token usage is shared across all users. Resets daily at midnight UTC.*"
    )


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def respond(message: str, chat_history: list[dict]):
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": "Thinking..."})
    yield chat_history

    result_container: dict = {}
    status_queue: "queue.Queue[str]" = queue.Queue()

    def _on_status(msg: str) -> None:
        status_queue.put(msg)

    def _run() -> None:
        try:
            result_container["result"] = ask(message, _on_status=_on_status)
        except APIError:
            result_container["api_error"] = True
        except Exception as exc:
            result_container["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        try:
            status = status_queue.get(timeout=0.3)
            chat_history[-1]["content"] = status
            yield chat_history
        except queue.Empty:
            if not thread.is_alive():
                break

    thread.join()

    if "api_error" in result_container:
        chat_history[-1]["content"] = (
            "The AI service is temporarily unavailable. "
            "This may be due to rate limits or API downtime. "
            "Please try again in a moment."
        )
        yield chat_history
        return

    if "error" in result_container:
        chat_history[-1]["content"] = (
            f"Something went wrong. Please try again.\n\n"
            f"Details: {result_container['error']!s}"
        )
        yield chat_history
        return

    result = result_container.get("result")
    if result is None:
        chat_history[-1]["content"] = "Something went wrong. Please try again."
        yield chat_history
        return

    answer = result["answer"]
    if result["low_confidence"]:
        answer += "\n\nLow confidence: Retrieved context scored near the relevance threshold."

    chat_history[-1]["content"] = answer
    yield chat_history


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="UI GreenMetric RAG Assistant v1.0") as app:
    gr.Markdown("# UI GreenMetric RAG Assistant v1.0")

    chatbot = gr.Chatbot(label="Chat")

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask a question about UI GreenMetric...",
            show_label=False,
            scale=4,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    budget_display = gr.Markdown(_budget_display() + "\n\n*Conversations may be logged for quality monitoring.*")

    # --- event bindings ---

    def _on_send(msg, hist):
        for chat_out in respond(msg, hist):
            yield chat_out, _budget_display(), ""

    send_btn.click(
        fn=_on_send,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, budget_display, msg_input],
    )

    msg_input.submit(
        fn=_on_send,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, budget_display, msg_input],
    )


if __name__ == "__main__":
    app.launch()
