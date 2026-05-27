"""Pipeline orchestrator for the UI GreenMetric RAG system.

Wires the router, retriever, and generator together into a single
end-to-end ``ask()`` entry point.  Manages conversation memory with
a configurable chat limit and handles the ``"none"`` route before any
expensive LLM or embedding calls are made.
"""

from src.router import route
from src.retriever import retrieve
from src.generator import generate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(
    query: str,
    *,
    conversation_history: list[dict] | None = None,
    max_turns: int = 7,
) -> dict:
    """Run the full RAG pipeline: route → retrieve → generate.

    Parameters:
        query:                The user's question.
        conversation_history: Prior user/assistant message pairs.  Each
                              dict has ``"role"`` (``"user"`` or
                              ``"assistant"``) and ``"content"``.  The
                              list is truncated to *max_turns* * 2 before
                              being passed to downstream modules and
                              returned in the result.
        max_turns:            Maximum number of conversation turns to
                              retain (default 7, i.e. 14 messages).
                              Excess messages are dropped FIFO.

    Returns:
        dict with keys:

        * ``"answer"`` — the generated answer string.  For ``"none"``
          routes this is an immediate polite refusal; no LLM or embedding
          calls are made.
        * ``"route"`` — the dict returned by :func:`router.route`.
        * ``"retrieved"`` — the number of chunks returned by
          :func:`retriever.retrieve` (0 for ``"none"``).
        * ``"contexts"`` — list of retrieved chunk content strings,
          suitable for RAGAS evaluation.  Empty list for ``"none"``.
        * ``"low_confidence"`` — ``True`` when the top retrieved chunk
          exceeded the 0.6 cosine-distance warning threshold.
        * ``"conversation_history"`` — the updated message list capped
          at *max_turns* * 2, ready for the next ``ask()`` call.
    """
    history = _manage_history(conversation_history, max_turns)

    route_result = route(query, conversation_history=history)

    if route_result["source"] == "none":
        answer = "I don't have the required information to answer this question."
        history = _append_turn(history, query, answer, max_turns)
        return {
            "answer": answer,
            "route": route_result,
            "retrieved": 0,
            "contexts": [],
            "low_confidence": False,
            "conversation_history": history,
        }

    context = retrieve(query, route_result)

    answer = generate(
        query,
        context,
        conversation_history=history,
        query_type=route_result["query_type"],
    )

    low_confidence = (
        bool(context)
        and context[0]["distance"] > 0.6
        and route_result["query_type"] != "aggregate"
    )

    history = _append_turn(history, query, answer, max_turns)

    return {
        "answer": answer,
        "route": route_result,
        "retrieved": len(context),
        "contexts": [c["content"] for c in context],
        "low_confidence": low_confidence,
        "conversation_history": history,
    }


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

def _manage_history(
    history: list[dict] | None, max_turns: int
) -> list[dict]:
    """Truncate conversation history to at most *max_turns* * 2 messages."""
    if not history:
        return []
    return history[-(max_turns * 2):]


def _append_turn(
    history: list[dict], query: str, answer: str, max_turns: int
) -> list[dict]:
    """Append a user/assistant turn and enforce the message cap."""
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    return history[-(max_turns * 2):]


# ---------------------------------------------------------------------------
# Budget tracking (v1.0 placeholders)
# ---------------------------------------------------------------------------

def _track_budget(call_type: str, **kwargs) -> None:
    """Placeholder — track token usage per LLM call.  Wired in v1.0."""
    pass


def _remaining_budget() -> bool:
    """Placeholder — check if budget allows another call.  Always True."""
    return True
