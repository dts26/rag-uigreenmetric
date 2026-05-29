"""Pipeline orchestrator for the UI GreenMetric RAG system.

Wires the router, retriever, RAG Fusion paraphrasing, and generator
into a single end-to-end ``ask()`` entry point.
"""

import os
import time

from src.router import route, paraphrase
from src.retriever import retrieve, retrieve_multi
from src.generator import generate

_RERANK_ENABLED = os.getenv("RAG_RERANK", "0") == "1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(query: str) -> dict:
    """Run the full RAG pipeline.

    Flow:
        source == "none"
          → immediate polite refusal (no LLM / embedding calls).
        query_type == "aggregate"
          → fetch all chunks via exact metadata match;
            skip paraphrase and reranker.
        everything else
          → paraphrase (3 variants) → multi-query retrieval
            → RRF merge → (optional reranker) → top 5.

    Parameters:
        query:  The user's question.

    Returns:
        dict with keys:

        * ``"answer"`` — the generated answer string.
        * ``"route"`` — the dict returned by :func:`router.route`.
        * ``"retrieved"`` — the number of chunks after retrieval.
        * ``"contexts"`` — list of retrieved chunk content strings.
        * ``"low_confidence"`` — ``True`` when the top chunk exceeded
          the 0.6 cosine-distance warning threshold.
        * ``"rerank_ms"`` — milliseconds spent in the reranker
          (0.0 when skipped).
    """
    route_result = route(query)

    # ── none ────────────────────────────────────────────────────────────
    if route_result["source"] == "none":
        return {
            "answer": "I don't have the required information to answer this question.",
            "route": route_result,
            "retrieved": 0,
            "contexts": [],
            "low_confidence": False,
            "rerank_ms": 0.0,
        }

    # ── aggregate ───────────────────────────────────────────────────────
    if route_result["query_type"] == "aggregate":
        context = retrieve(query, route_result)
        answer = generate(query, context, query_type="aggregate")
        return {
            "answer": answer,
            "route": route_result,
            "retrieved": len(context),
            "contexts": [c["content"] for c in context],
            "low_confidence": False,
            "rerank_ms": 0.0,
        }

    # ── lookup / both → RAG Fusion ──────────────────────────────────────
    variants = paraphrase(query)
    queries = [query] + variants  # 1 original + up to 3 paraphrases

    context = retrieve_multi(queries, route_result)

    rerank_ms = 0.0
    if _RERANK_ENABLED and context:
        from src.reranker import rerank
        t0 = time.perf_counter()
        context = rerank(query, context, top_n=5)
        rerank_ms = (time.perf_counter() - t0) * 1000

    answer = generate(query, context, query_type=route_result["query_type"])

    low_confidence = (
        bool(context)
        and context[0].get("distance", 0.0) > 0.6
    )

    return {
        "answer": answer,
        "route": route_result,
        "retrieved": len(context),
        "contexts": [c["content"] for c in context],
        "low_confidence": low_confidence,
        "rerank_ms": rerank_ms,
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
