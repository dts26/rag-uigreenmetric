"""Pipeline orchestrator for the UI GreenMetric RAG system.

Wires the router, retriever, RAG Fusion paraphrasing, and generator
into a single end-to-end ``ask()`` entry point.
"""

import os
import time

from src.router import route, paraphrase
from src.retriever import retrieve, retrieve_multi
from src.generator import generate
from src.budget import BudgetManager, MemoryBudgetStore, HFBudgetStore
from src.conversation import log_conversation, get_logger


def _flush_logs() -> None:
    logger = get_logger()
    if logger:
        logger.flush()

_RERANK_ENABLED = os.getenv("RAG_RERANK", "0") == "1"

_BUDGET_BLOCKED_RESPONSE = {
    "answer": "Daily token budget reached. Please try again tomorrow.",
    "route": {"source": "none", "csv_source": None, "query_type": "lookup"},
    "retrieved": 0,
    "contexts": [],
    "low_confidence": False,
    "rerank_ms": 0.0,
}

# Budget: use HF dataset store if repo configured, else in-memory
_budget_repo = os.getenv("RAG_BUDGET_REPO", "")
_budget = BudgetManager(
    store=HFBudgetStore(_budget_repo) if _budget_repo else MemoryBudgetStore(),
    daily_cap=int(os.getenv("RAG_BUDGET_TOKENS", "200000")),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(
    query: str,
    *,
    _route_result: dict | None = None,
    _fusion_queries: list[str] | None = None,
) -> dict:
    """Run the full RAG pipeline.

    Flow:
        source == "none"
          → immediate polite refusal (no LLM / embedding calls).
        query_type == "aggregate"
          → fetch all chunks via exact metadata match;
            skip paraphrase and reranker.
        everything else
          → paraphrase (3 variants) → multi-query retrieval
            → RRF merge → (optional reranker) → top 7.

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
    # ── budget guard ─────────────────────────────────────────────────
    if _budget.exceeded():
        return _BUDGET_BLOCKED_RESPONSE

    if _route_result is not None:
        route_result = _route_result  # pre-computed, no LLM call
    else:
        route_result, route_tokens = route(query)
        _budget.track(route_tokens)

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
        if _budget.exceeded():
            return _BUDGET_BLOCKED_RESPONSE
        answer, gen_tokens = generate(query, context, query_type="aggregate")
        _budget.track(gen_tokens)
        log_conversation(query, answer, route_result,
                         [c["content"] for c in context], gen_tokens)
        _flush_logs()
        return {
            "answer": answer,
            "route": route_result,
            "retrieved": len(context),
            "contexts": [c["content"] for c in context],
            "low_confidence": False,
            "rerank_ms": 0.0,
        }

    # ── lookup / both → RAG Fusion ──────────────────────────────────────
    if _fusion_queries is not None:
        queries = _fusion_queries
    else:
        variants, para_tokens = paraphrase(query)
        _budget.track(para_tokens)
        queries = [query] + variants

    context = retrieve_multi(queries, route_result)

    rerank_ms = 0.0
    if _RERANK_ENABLED and context:
        from src.reranker import rerank
        t0 = time.perf_counter()
        context = rerank(query, context, top_n=5)
        rerank_ms = (time.perf_counter() - t0) * 1000
    else:
        context = context[:7]

    if _budget.exceeded():
        return _BUDGET_BLOCKED_RESPONSE

    answer, gen_tokens = generate(query, context, query_type=route_result["query_type"])
    _budget.track(gen_tokens)

    low_confidence = (
        bool(context)
        and context[0].get("distance", 0.0) > 0.6
    )

    log_conversation(query, answer, route_result,
                     [c["content"] for c in context], gen_tokens)
    _flush_logs()

    return {
        "answer": answer,
        "route": route_result,
        "retrieved": len(context),
        "contexts": [c["content"] for c in context],
        "low_confidence": low_confidence,
        "rerank_ms": rerank_ms,
    }
