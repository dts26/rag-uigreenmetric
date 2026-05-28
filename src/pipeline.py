"""Pipeline orchestrator for the UI GreenMetric RAG system.

Wires the router, retriever, and generator together into a single
end-to-end ``ask()`` entry point.
"""

from src.router import route
from src.retriever import retrieve
from src.generator import generate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(query: str) -> dict:
    """Run the full RAG pipeline: route → retrieve → generate.

    Parameters:
        query:  The user's question.

    Returns:
        dict with keys:

        * ``"answer"`` — the generated answer string.  For ``"none"``
          routes this is an immediate polite refusal; no LLM or embedding
          calls are made.
        * ``"route"`` — the dict returned by :func:`router.route`.
        * ``"retrieved"`` — the number of chunks returned by
          :func:`retriever.retrieve` (0 for ``"none"``).
        * ``"contexts"`` — list of retrieved chunk content strings,
          suitable for evaluation.  Empty list for ``"none"``.
        * ``"low_confidence"`` — ``True`` when the top retrieved chunk
          exceeded the 0.6 cosine-distance warning threshold.
    """
    route_result = route(query)

    if route_result["source"] == "none":
        return {
            "answer": "I don't have the required information to answer this question.",
            "route": route_result,
            "retrieved": 0,
            "contexts": [],
            "low_confidence": False,
        }

    context = retrieve(query, route_result)

    answer = generate(query, context, query_type=route_result["query_type"])

    low_confidence = (
        bool(context)
        and context[0]["distance"] > 0.6
        and route_result["query_type"] != "aggregate"
    )

    return {
        "answer": answer,
        "route": route_result,
        "retrieved": len(context),
        "contexts": [c["content"] for c in context],
        "low_confidence": low_confidence,
    }


# ---------------------------------------------------------------------------
# Budget tracking (v1.0 placeholders)
# ---------------------------------------------------------------------------

def _track_budget(call_type: str, **kwargs) -> None:
    """Placeholder — track token usage per LLM call.  Wired in v1.0."""
    pass


def _remaining_budget() -> bool:
    """Placeholder — check if budget allows another call.  Always True."""
    return True
