"""UI GreenMetric RAG Assistant v0.5 — Gradio web interface."""

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
        "❌ DEEPSEEK_API_KEY not found. "
        "Set it in your .env file or as an environment variable."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _debug_info(result: dict, session: dict) -> str:
    """Format one turn as a compact debug entry."""
    route = result["route"]
    return (
        f"[T{session['turn_count']}] "
        f"{route['source']} | {route.get('csv_source', '-')} | {route.get('query_type', '-')} "
        f"| {result['retrieved']} chunks | LC: {result['low_confidence']}\n"
        "──────────────────────────────────────────────────────────────"
    )


# ---------------------------------------------------------------------------
# Budget stubs (v1.0 — always pass)
# ---------------------------------------------------------------------------

def _daily_cap_ok() -> bool:
    """Check if the daily token budget has been exceeded.  Stub — always True."""
    return True


def _circuit_breaker_ok() -> bool:
    """Check if a hard circuit breaker has been tripped.  Stub — always True."""
    return True


def _session_rate_ok() -> bool:
    """Check if the session is rate-limited.  Stub — always True."""
    return True


def _budget_blocked() -> str | None:
    """Return a user-facing message if any budget limit is hit, else None."""
    checks = [
        (_daily_cap_ok, "📊 **Daily token budget reached.** Please try again tomorrow."),
        (_circuit_breaker_ok, "🔒 **Service temporarily paused.** Maintenance in progress."),
        (_session_rate_ok, "⏳ **Rate limit reached.** Please wait a moment before sending another message."),
    ]
    for check_fn, msg in checks:
        if not check_fn():
            return msg
    return None


def _debug_blocked(message: str) -> str:
    """Format a budget-block debug summary."""
    return f"🚫 **Budget blocked:** {message}"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def reset_session():
    """Return a fresh empty chat and session state."""
    return [], {"conversation_history": [], "turn_count": 0}


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def respond(message: str, chat_history: list[dict], session: dict):
    """Handle one user message: append to chat, call pipeline, yield answer."""

    turn_count = session["turn_count"]

    if turn_count >= MAX_TURNS:
        limit_msg = (
            "💡 **Session limit reached (7 turns).** "
            "Click 'Reset Session' to start a new conversation."
        )
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": limit_msg})
        yield chat_history, session, ""
        return

    blocked = _budget_blocked()
    if blocked:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": blocked})
        yield chat_history, session, _debug_blocked(blocked)
        return

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": "Thinking..."})
    yield chat_history, session, ""

    try:
        result = ask(message, conversation_history=session["conversation_history"])
    except APIError:
        chat_history[-1]["content"] = (
            "⚠️ **The AI service is temporarily unavailable.** "
            "This may be due to rate limits or API downtime. "
            "Please try again in a moment."
        )
        yield chat_history, session, ""
        return
    except Exception as exc:
        chat_history[-1]["content"] = (
            f"⚠️ **Something went wrong.** Please try again.\n\n"
            f"*Details: {exc!s}*"
        )
        yield chat_history, session, ""
        return

    answer = result["answer"]
    if result["low_confidence"]:
        answer += "\n\n⚠️ **Low confidence:** Retrieved context scored near the relevance threshold."

    chat_history[-1]["content"] = answer

    session = {
        "conversation_history": result["conversation_history"],
        "turn_count": turn_count + 1,
    }

    debug = _debug_info(result, session)

    yield chat_history, session, debug


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
        reset_btn = gr.Button("🔄 Reset Session", size="sm")

    session_state = gr.State({"conversation_history": [], "turn_count": 0})

    with gr.Accordion("Debug Info", open=False):
        debug_panel = gr.Textbox(
            value="*No query submitted yet.*",
            lines=14,
            label="",
            interactive=False,
            show_copy_button=True,
        )

    debug_state = gr.State("*No query submitted yet.*")

    # --- event bindings ---

    def _on_send(msg, hist, sess, debug_log):
        for chat_out, sess_out, debug_out in respond(msg, hist, sess):
            turn_display = f"{sess_out['turn_count']} / {MAX_TURNS} messages used"
            if debug_out:
                debug_log = debug_out + "\n" + debug_log
            yield chat_out, turn_display, debug_log, sess_out, ""

    send_btn.click(
        fn=_on_send,
        inputs=[msg_input, chatbot, session_state, debug_state],
        outputs=[chatbot, turn_counter, debug_panel, session_state, msg_input],
    )

    msg_input.submit(
        fn=_on_send,
        inputs=[msg_input, chatbot, session_state, debug_state],
        outputs=[chatbot, turn_counter, debug_panel, session_state, msg_input],
    )

    reset_btn.click(
        fn=lambda: ([], "0 / 7 messages used", "*No query submitted yet.*",
                     {"conversation_history": [], "turn_count": 0},
                     "*No query submitted yet.*", ""),
        inputs=[],
        outputs=[chatbot, turn_counter, debug_panel, session_state, debug_state, msg_input],
    )


if __name__ == "__main__":
    app.launch()
