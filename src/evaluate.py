"""
Evaluation pipeline for the UI GreenMetric RAG system.

Loads test cases from ``test_cases/test_cases.xlsx``, runs each through
:func:`src.pipeline.ask`, computes RAGAS metrics (Faithfulness, Context
Recall, Response Relevancy, Answer Correctness) for in-domain questions,
validates refusals for out-of-domain questions, and tracks router
accuracy.

Usage::

    python -m src.evaluate
"""

import os
import sys
import math
from dataclasses import dataclass, field

# ── RAGAS import fix ─────────────────────────────────────────────────────────
# langchain_community 0.4.x removed chat_models.vertexai; RAGAS 0.4.3 still
# imports it.  Remap to the standalone package.
import langchain_google_vertexai  # noqa: E402

sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

import pandas as pd  # noqa: E402
from openai import OpenAI  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from ragas.llms import llm_factory  # noqa: E402
from ragas import evaluate, EvaluationDataset, SingleTurnSample  # noqa: E402
# NOTE: using deprecated ragas.metrics import — the collections variants
# require llm/embeddings ctor args in 0.4.3.  Stable for now.
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    ContextRecall,
    ResponseRelevancy,
    AnswerCorrectness,
)

from src.pipeline import ask  # noqa: E402

load_dotenv()

# ── LLM client for RAGAS scoring ─────────────────────────────────────────────

_ragas_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# Use deepseek-chat (V3) for RAGAS scoring — v4 models burn the output
# token budget on reasoning_content, causing instructor to retry until
# max_retries and fail with an empty result.
RAGAS_LLM = llm_factory("deepseek-chat", provider="openai", client=_ragas_client)


# ── Embeddings for RAGAS ─────────────────────────────────────────────────────
# evaluate() defaults to OpenAIEmbeddings which fails with the DeepSeek API
# key (wrong base URL) and also hits aembed_text / embed_text async-vs-sync
# mismatch under nest_asyncio.  Provide our own wrapper around the same
# SentenceTransformer model already used by the pipeline.

import asyncio  # noqa: E402
import typing as t  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402

_EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


class _STEmbedding(BaseRagasEmbedding):
    """BaseRagasEmbedding backed by a SentenceTransformer model.

    Also exposes ``embed_query`` / ``embed_documents`` for LangChain
    compatibility (needed by ResponseRelevancy).
    """

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:  # type: ignore[override]
        return _EMBED_MODEL.encode(text, show_progress_bar=False).tolist()

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        # Delegate to sync version via thread — SentenceTransformer is CPU-bound.
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    # LangChain-compatible aliases — ResponseRelevancy calls embed_query.
    def embed_query(self, text: str) -> t.List[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return self.embed_texts(texts)


RAGAS_EMBEDDINGS = _STEmbedding()


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """One test case after evaluation."""

    idx: int
    question: str
    ground_truth: str
    expected_source: str
    notes: str = ""

    # ── captured from ask() ──
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    actual_source: str = ""
    actual_csv_source: str | None = None
    actual_query_type: str = ""
    low_confidence: bool = False

    # ── classification ──
    is_none_case: bool = False
    refusal_correct: bool | None = None  # None for non-none cases
    route_correct: bool = False

    # ── RAGAS scores (0..1, float) ──
    faithfulness: float | None = None
    context_recall: float | None = None
    response_relevancy: float | None = None
    answer_correctness: float | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Test-case loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_cases(path: str = "test_cases/test_cases.xlsx") -> list[EvalResult]:
    """Load the xlsx test set into :class:`EvalResult` rows."""
    df = pd.read_excel(path)
    results: list[EvalResult] = []
    for i, row in df.iterrows():
        notes = row.get("notes")
        results.append(
            EvalResult(
                idx=i,
                question=row["question"],
                ground_truth=row["ground_truth"],
                expected_source=row["source"],
                notes=str(notes) if pd.notna(notes) else "",
            )
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Route normalisation & comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_expected(raw: str) -> tuple[str, str | None]:
    """Convert a test-case ``source`` string into router-format keys.

    >>> _normalise_expected("none")
    ('none', None)
    >>> _normalise_expected("pdf")
    ('pdf', None)
    >>> _normalise_expected("csv_appendix1")
    ('csv', 'csv_appendix1')
    >>> _normalise_expected("both_pdf_csv_table4")
    ('both', 'csv_table4')
    """
    if raw == "none":
        return ("none", None)
    if raw == "pdf":
        return ("pdf", None)
    if raw.startswith("both_pdf_"):
        # "both_pdf_csv_appendix1" → ("both", "csv_appendix1")
        return ("both", raw[len("both_pdf_"):])
    if raw.startswith("csv_"):
        return ("csv", raw)
    return ("unknown", None)


def check_route(expected_source: str, route: dict) -> bool:
    """Return ``True`` when *route* matches the expected data source."""
    want_source, want_csv = _normalise_expected(expected_source)
    got_source = route["source"]
    got_csv = route.get("csv_source")

    if want_source != got_source:
        return False
    if want_csv is not None and want_csv != got_csv:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Refusal detection
# ═══════════════════════════════════════════════════════════════════════════════

REFUSAL_PATTERNS: list[str] = [
    "i don't have the required information",
    "sorry, i don't know",
    "sorry, i cannot",
    "not available in our document",
    "the document does not",
    "the guideline document does not",
    "does not contain",
    "information regarding your question is not available",
    "information is not available",
    "no information",
    "not provided",
    "not found",
    "cannot answer",
    "outside the scope",
    "please refer to",
    "not within the scope",
]


def _is_refusal(answer: str) -> bool:
    """Heuristic refusal check — ``True`` when *answer* declines to answer."""
    lowered = answer.lower()
    # Strong refusal: starts with the pipeline's stock response
    if lowered.startswith("i don't have the required information"):
        return True
    # Match patterns
    for pat in REFUSAL_PATTERNS:
        if pat in lowered:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# RAGAS batch scoring
# ═══════════════════════════════════════════════════════════════════════════════

RAGAS_METRICS = [
    Faithfulness(),
    ContextRecall(),
    ResponseRelevancy(),
    AnswerCorrectness(),
]

_SCORE_FIELDS = (
    "faithfulness",
    "context_recall",
    "response_relevancy",
    "answer_correctness",
)


def _run_ragas(results: list[EvalResult]) -> list[EvalResult]:
    """Score every non-none result with RAGAS in a single batch.

    Only *Faithfulness*, *ContextRecall*, and *AnswerCorrectness* are
    scored via LLM.  *ResponseRelevancy* requires an embeddings provider
    compatible with RAGAS (OpenAIEmbeddings) which does not work with
    the DeepSeek API key—it is skipped and left as ``None`` for all
    results.
    """
    non_none = [r for r in results if not r.is_none_case and r.contexts]
    if not non_none:
        return results

    # All four metrics are used now — we provide our own SentenceTransformer
    # embeddings (RAGAS_EMBEDDINGS) so ResponseRelevancy and ContextRecall work.
    llm_metrics = list(RAGAS_METRICS)

    samples: list[SingleTurnSample] = []
    for r in non_none:
        samples.append(
            SingleTurnSample(
                user_input=r.question,
                response=r.answer,
                retrieved_contexts=r.contexts,
                reference=r.ground_truth,
            )
        )

    dataset = EvaluationDataset(samples=samples)
    eval_out = evaluate(
        dataset=dataset,
        metrics=llm_metrics,
        llm=RAGAS_LLM,
        embeddings=RAGAS_EMBEDDINGS,
    )

    scored = eval_out.to_pandas()
    # Map RAGAS column names to our EvalResult field names.
    # RAGAS 0.4.3 column names differ from the legacy metric names:
    # 'answer_relevancy' is what ResponseRelevancy writes.
    _COLUMN_MAP = {
        "faithfulness": "faithfulness",
        "context_recall": "context_recall",
        "answer_relevancy": "response_relevancy",
        "answer_correctness": "answer_correctness",
    }
    for i, r in enumerate(non_none):
        row = scored.iloc[i]
        for col, field in _COLUMN_MAP.items():
            if col in scored.columns:
                val = row[col]
                setattr(r, field, float(val) if pd.notna(val) else None)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_stats(results: list[EvalResult]) -> dict:
    """Derive aggregate scores, per-source breakdowns, and weakest-5."""
    non_none = [r for r in results if not r.is_none_case]
    none_cases = [r for r in results if r.is_none_case]

    # ── Aggregate RAGAS ──
    agg: dict[str, float] = {}
    for f in _SCORE_FIELDS:
        vals = [getattr(r, f) for r in non_none if getattr(r, f) is not None]
        agg[f] = sum(vals) / len(vals) if vals else math.nan

    # ── Per-source breakdown ──
    per_source: dict[str, dict] = {}
    for r in non_none:
        src = r.expected_source
        bucket = per_source.setdefault(src, {f: [] for f in _SCORE_FIELDS})
        for f in _SCORE_FIELDS:
            v = getattr(r, f)
            if v is not None:
                bucket[f].append(v)

    for src in per_source:
        bucket = per_source[src]
        for f in _SCORE_FIELDS:
            vals = bucket[f]
            bucket[f] = sum(vals) / len(vals) if vals else math.nan
        bucket["count"] = sum(
            1 for r in non_none if r.expected_source == src
        )

    # ── None cases ──
    none_correct = sum(1 for r in none_cases if r.refusal_correct is True)
    none_total = len(none_cases)
    none_acc = none_correct / none_total if none_total else 1.0

    # ── Router accuracy ──
    total = len(results)
    route_correct = sum(1 for r in results if r.route_correct)

    # ── Weakest 5 ──
    # Only rank on metrics that actually scored (Faithfulness is the only
    # one that works with DeepSeek; AnswerCorrectness returns NaN).
    _scored_fields = [
        f for f in _SCORE_FIELDS
        if any(getattr(r, f) is not None for r in non_none)
    ] or [f for f in _SCORE_FIELDS]  # fallback: all fields (defensive)

    scored = [
        r
        for r in non_none
        if all(getattr(r, f) is not None for f in _scored_fields)
    ]
    if _scored_fields:
        scored.sort(
            key=lambda r: sum(getattr(r, f) for f in _scored_fields) / len(_scored_fields),
        )
    weakest = scored[:5]

    return {
        "aggregate": agg,
        "per_source": per_source,
        "none_accuracy": none_acc,
        "none_correct": none_correct,
        "none_total": none_total,
        "router_accuracy": route_correct / total if total else 0.0,
        "router_correct": route_correct,
        "router_total": total,
        "weakest": weakest,
        "total_cases": total,
        "non_none_count": len(non_none),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report formatting
# ═══════════════════════════════════════════════════════════════════════════════

def _report(results: list[EvalResult], stats: dict) -> None:
    """Print a formatted evaluation report to stdout."""
    SEP = "=" * 80
    LABELS = {
        "faithfulness": "Faithfulness",
        "context_recall": "Context Recall",
        "response_relevancy": "Response Relevancy",
        "answer_correctness": "Answer Correctness",
    }

    print(f"\n{SEP}")
    print("  UI GREENMETRIC RAG SYSTEM — EVALUATION REPORT")
    print(f"{SEP}")
    print(
        f"  Cases: {stats['total_cases']} total  "
        f"({stats['non_none_count']} RAGAS + {stats['none_total']} none)"
    )
    print()

    # ── 1. Aggregate ──
    print("━━━ 1. AGGREGATE RAGAS SCORES ━━━")
    print(f"  {'Metric':<25} {'Score':>8}")
    print(f"  {'-' * 33}")
    for f in _SCORE_FIELDS:
        v = stats["aggregate"][f]
        print(f"  {LABELS[f]:<25} {v:>8.4f}" if not math.isnan(v) else f"  {LABELS[f]:<25} {'N/A':>8}")
    print()

    # ── 2. Per-source breakdown ──
    print("━━━ 2. PER-SOURCE BREAKDOWN ━━━")
    header = f"  {'Source':<30} {'#':>3}"
    for f in _SCORE_FIELDS:
        header += f" {LABELS[f][:7]:>8}"
    print(header)
    print(f"  {'-' * 30} {'-' * 3} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for src in sorted(stats["per_source"].keys()):
        ps = stats["per_source"][src]
        row = f"  {src:<30} {ps['count']:>3}"
        for f in _SCORE_FIELDS:
            v = ps[f]
            row += f" {v:>8.4f}" if not math.isnan(v) else "     N/A"
        print(row)
    print()

    # ── 3. Router accuracy ──
    print("━━━ 3. ROUTER ACCURACY ━━━")
    print(
        f"  Correct: {stats['router_correct']}/{stats['router_total']}  "
        f"({stats['router_accuracy']:.1%})"
    )
    # Show misrouted cases
    misrouted = [r for r in results if not r.route_correct]
    if misrouted:
        print(f"\n  Misrouted ({len(misrouted)}):")
        for r in misrouted:
            print(
                f"    #{r.idx}  expected={r.expected_source}  "
                f"got=source:{r.actual_source}"
                + (f"/{r.actual_csv_source}" if r.actual_csv_source else "")
            )
            print(f"         Q: {r.question[:90]}...")
    print()

    # ── 4. None cases ──
    print("━━━ 4. NONE CASES (Refusal Check) ━━━")
    print(
        f"  Correct refusals: {stats['none_correct']}/{stats['none_total']}  "
        f"({stats['none_accuracy']:.1%})"
    )
    none_cases = [r for r in results if r.is_none_case]
    for r in none_cases:
        mark = "✓" if r.refusal_correct else "✗"
        route_mark = "✓" if r.route_correct else "✗"
        print(f"  [{mark}] Route {route_mark} | {r.question[:90]}...")
        if not r.refusal_correct:
            print(f"         Answer: {r.answer[:150]}...")
    print()

    # ── 5. Weakest 5 ──
    print("━━━ 5. WEAKEST 5 (by average RAGAS score) ━━━")
    for i, r in enumerate(stats["weakest"]):
        avg = (
            sum(getattr(r, f) for f in _SCORE_FIELDS) / len(_SCORE_FIELDS)
            if all(getattr(r, f) is not None for f in _SCORE_FIELDS)
            else 0
        )
        print(f"  #{i + 1}  avg={avg:.4f}  Q: {r.question[:90]}...")
        for f in _SCORE_FIELDS:
            v = getattr(r, f)
            print(f"         {LABELS[f]:<22} {v:.4f}" if v is not None else f"         {LABELS[f]:<22} N/A")
        print(f"         Answer:        {r.answer[:120]}...")
        print(f"         Ground truth:  {r.ground_truth[:120]}...")
        print()

    # ── 6. Per-question table ──
    print("━━━ 6. PER-QUESTION RESULTS ━━━")
    header = (
        f"  {'#':>3} {'Question':<48} {'F':>6} {'CR':>6} {'RR':>6} {'AC':>6} {'Route':>6}"
    )
    print(header)
    print(f"  {'-' * 86}")
    for r in results:
        q = r.question[:45] + "..." if len(r.question) > 48 else r.question.ljust(48)
        route_mark = "✓" if r.route_correct else "✗"
        if r.is_none_case:
            ref = "✓" if r.refusal_correct else "✗"
            print(f"  {r.idx:>3} {q} {'NONE' + ref:>6} {'':>6} {'':>6} {'':>6} {route_mark:>6}")
        else:
            f_val = f"{r.faithfulness:.2f}" if r.faithfulness is not None else "  N/A"
            cr_val = f"{r.context_recall:.2f}" if r.context_recall is not None else "  N/A"
            rr_val = f"{r.response_relevancy:.2f}" if r.response_relevancy is not None else "  N/A"
            ac_val = f"{r.answer_correctness:.2f}" if r.answer_correctness is not None else "  N/A"
            print(f"  {r.idx:>3} {q} {f_val:>6} {cr_val:>6} {rr_val:>6} {ac_val:>6} {route_mark:>6}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    file_path: str = "test_cases/test_cases.xlsx",
) -> tuple[list[EvalResult], dict]:
    """Run the complete evaluation pipeline.

    1. Load test cases.
    2. Run each question through ``ask()``.
    3. Batch-score with RAGAS (skip ``none`` cases).
    4. Compute aggregate & per-source stats.
    5. Print the report.

    Returns:
        Tuple of (per-question results, aggregate stats dict).
    """
    print("═══ Loading test cases ═══")
    results = load_test_cases(file_path)
    print(f"Loaded {len(results)} cases.")

    print("\n═══ Running pipeline ═══")
    for i, r in enumerate(results):
        print(f"  [{i + 1:>2}/{len(results)}] {r.question[:85]}...")

        try:
            out = ask(r.question)
        except Exception as exc:
            print(f"         ⚠  ask() failed: {exc}")
            r.answer = f"[ERROR: {exc}]"
            r.contexts = []
            r.actual_source = "none"
            r.route_correct = False
            if r.is_none_case:
                r.refusal_correct = False
            continue

        r.answer = out["answer"]
        r.contexts = out["contexts"]
        r.actual_source = out["route"]["source"]
        r.actual_csv_source = out["route"].get("csv_source")
        r.actual_query_type = out["route"].get("query_type", "lookup")
        r.low_confidence = out["low_confidence"]

        # None-case classification & refusal check
        if r.expected_source == "none":
            r.is_none_case = True
            r.refusal_correct = _is_refusal(r.answer)

        # Route accuracy
        r.route_correct = check_route(r.expected_source, out["route"])

    # Batch RAGAS scoring
    print("\n═══ Scoring with RAGAS ═══")
    try:
        results = _run_ragas(results)
    except Exception as exc:
        print(f"⚠  RAGAS scoring failed: {exc}")
        print("   Continuing with report (RAGAS scores will be N/A).")

    # Stats & report
    stats = _compute_stats(results)
    _report(results, stats)

    return results, stats


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
