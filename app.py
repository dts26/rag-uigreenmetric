"""UI GreenMetric RAG Assistant v0.5.1 — Gradio web interface."""

import os
import gradio as gr
from openai import APIError
from src.pipeline import ask

MAX_TURNS = 7

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
# Budget stubs (v1.0 — always pass)
# ---------------------------------------------------------------------------

def _daily_cap_ok() -> bool:
    return True


def _circuit_breaker_ok() -> bool:
    return True


def _session_rate_ok() -> bool:
    return True


def _budget_blocked() -> str | None:
    checks = [
        (_daily_cap_ok, "Daily token budget reached. Please try again tomorrow."),
        (_circuit_breaker_ok, "Service temporarily paused. Maintenance in progress."),
        (_session_rate_ok, "Rate limit reached. Please wait a moment before sending another message."),
    ]
    for check_fn, msg in checks:
        if not check_fn():
            return msg
    return None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def reset_session():
    return [], {"turn_count": 0}


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def respond(message: str, chat_history: list[dict], session: dict):
    turn_count = session["turn_count"]

    if turn_count >= MAX_TURNS:
        limit_msg = (
            "Session limit reached (7 turns). "
            "Click 'Reset Session' to start a new conversation."
        )
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": limit_msg})
        yield chat_history, session
        return

    blocked = _budget_blocked()
    if blocked:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": blocked})
        yield chat_history, session
        return

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": "Thinking..."})
    yield chat_history, session

    try:
        result = ask(message)
    except APIError:
        chat_history[-1]["content"] = (
            "The AI service is temporarily unavailable. "
            "This may be due to rate limits or API downtime. "
            "Please try again in a moment."
        )
        yield chat_history, session
        return
    except Exception as exc:
        chat_history[-1]["content"] = (
            f"Something went wrong. Please try again.\n\n"
            f"Details: {exc!s}"
        )
        yield chat_history, session
        return

    answer = result["answer"]
    if result["low_confidence"]:
        answer += "\n\nLow confidence: Retrieved context scored near the relevance threshold."

    chat_history[-1]["content"] = answer
    session = {"turn_count": turn_count + 1}

    yield chat_history, session


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="UI GreenMetric RAG Assistant v0.5") as app:
    gr.Markdown("# UI GreenMetric RAG Assistant v0.5")

    chatbot = gr.Chatbot(label="Chat")

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask a question about UI GreenMetric...",
            show_label=False,
            scale=4,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        turn_counter = gr.Markdown("0 / 7 messages used")
        reset_btn = gr.Button("Reset Session", size="sm")

    session_state = gr.State({"turn_count": 0})

    # --- event bindings ---

    def _on_send(msg, hist, sess):
        for chat_out, sess_out in respond(msg, hist, sess):
            turn_display = f"{sess_out['turn_count']} / {MAX_TURNS} messages used"
            yield chat_out, turn_display, sess_out, ""

    send_btn.click(
        fn=_on_send,
        inputs=[msg_input, chatbot, session_state],
        outputs=[chatbot, turn_counter, session_state, msg_input],
    )

    msg_input.submit(
        fn=_on_send,
        inputs=[msg_input, chatbot, session_state],
        outputs=[chatbot, turn_counter, session_state, msg_input],
    )

    reset_btn.click(
        fn=lambda: ([], "0 / 7 messages used", {"turn_count": 0}, ""),
        inputs=[],
        outputs=[chatbot, turn_counter, session_state, msg_input],
    )


if __name__ == "__main__":
    app.launch()
