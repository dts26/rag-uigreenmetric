"""
Evaluation pipeline for the UI GreenMetric RAG system — powered by DeepEval.

v0.5.1 — replaces the RAGAS-based evaluate_ragas.py with DeepEval metrics:
  - Faithfulness          -> FaithfulnessMetric
  - Context Recall        -> ContextualRecallMetric
  - NDCG@K                -> ContextualPrecisionMetric
  - G-Eval                -> GEval (compares generated answer to ground truth, 0-1)
  - Router Accuracy       -> RouterAccuracyMetric (custom, checks route vs label)
"""

import glob
import os
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    BaseMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    GEval,
)
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate.configs import DisplayConfig
from src.pipeline import ask

load_dotenv()

# Save JSON results per run (avoids "no results_folder" warning)
os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", "./data")


# ---------------------------------------------------------------------------
# Custom LLM judge — wraps DeepSeek for DeepEval
# ---------------------------------------------------------------------------


class DeepSeekEvalLLM(DeepEvalBaseLLM):
    """Custom LLM judge backed by the DeepSeek API."""

    def __init__(self, model_name: str = "deepseek-v4-flash"):
        self.model_name = model_name
        self._client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def load_model(self):
        return self._client

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=8192,
            temperature=0.0,
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        import asyncio
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return f"DeepSeek {self.model_name}"


EVAL_MODEL = DeepSeekEvalLLM("deepseek-v4-flash")

# ---------------------------------------------------------------------------
# Router Accuracy — custom DeepEval metric
# ---------------------------------------------------------------------------


class RouterAccuracyMetric(BaseMetric):
    """Measures whether the pipeline router returned the expected source."""

    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.error = None

    @staticmethod
    def _compare(expected: str, route: dict) -> bool:
        if expected == "none":
            return route["source"] == "none"
        if expected == "pdf":
            return route["source"] == "pdf"
        if expected.startswith("both_pdf_"):
            csv = expected[len("both_pdf_"):]
            return route["source"] == "both" and route.get("csv_source") == csv
        if expected.startswith("csv_"):
            return route["source"] == "csv" and route.get("csv_source") == expected
        return False

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            expected = test_case.additional_metadata["expected_source"]
            route = test_case.additional_metadata["route"]
            self.score = 1.0 if self._compare(expected, route) else 0.0
            if self.include_reason:
                r_src = route["source"]
                r_csv = route.get("csv_source")
                actual = r_src + (f"/{r_csv}" if r_csv else "")
                self.reason = (
                    f"matched (expected={expected})"
                    if self.score == 1.0
                    else f"mismatch: expected={expected}, got={actual}"
                )
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Router Accuracy"


# ---------------------------------------------------------------------------
# RAG metrics
# ---------------------------------------------------------------------------

FAITHFULNESS = FaithfulnessMetric(
    threshold=0.7,
    model=EVAL_MODEL,
    include_reason=True,
)

CONTEXT_RECALL = ContextualRecallMetric(
    threshold=0.7,
    model=EVAL_MODEL,
    include_reason=True,
)

CONTEXT_PRECISION = ContextualPrecisionMetric(
    threshold=0.7,
    model=EVAL_MODEL,
    include_reason=True,
)

STRICT_CORRECTNESS = GEval(
    name="Strict Answer Correctness",
    criteria=(
        "Compare the 'actual_output' against the 'expected_output'. "
        "Evaluate how accurately the actual output captures the core facts, "
        "numeric thresholds, and logical constraints strictly defined in the "
        "expected output. Score on a continuous spectrum from 0 to 1, where 1 "
        "means a perfect factual match, and 0 means complete failure, "
        "contradiction, or an 'I don't know' fallback."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=EVAL_MODEL,
)

ROUTER_ACCURACY = RouterAccuracyMetric(threshold=0.5)

ALL_METRICS = [
    FAITHFULNESS,
    CONTEXT_RECALL,
    CONTEXT_PRECISION,
    STRICT_CORRECTNESS,
]

_METRIC_LABELS = {
    "Faithfulness": "F",
    "Contextual Recall": "CR",
    "Contextual Precision": "CP",
    "Strict Answer Correctness": "GE",
    "Router Accuracy": "RT",
}


# ---------------------------------------------------------------------------
# Test case loading
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """One row from the test case xlsx — raw data before running ask()."""

    idx: int
    question: str
    ground_truth: str
    expected_source: str
    notes: str = ""


def load_test_cases(path: str = "test_cases/test_cases.xlsx") -> list[TestCase]:
    """Load test cases from the Excel file."""
    df = pd.read_excel(path)
    cases: list[TestCase] = []
    for i, row in df.iterrows():
        cases.append(
            TestCase(
                idx=i,
                question=row["question"],
                ground_truth=row["ground_truth"],
                expected_source=row["source"],
                notes=str(row.get("notes", ""))
                if pd.notna(row.get("notes", ""))
                else "",
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------


def run_evaluation(path: str = "test_cases/test_cases.xlsx") -> None:
    test_cases = load_test_cases(path=path)
    deep_eval_cases: list[LLMTestCase] = []
    none_idxs: set[int] = set()
    case_route_marks: dict[int, bool] = {}
    case_contexts: dict[int, list[str]] = {}  # case.idx -> retrieved chunks
    case_answers: dict[int, str] = {}         # case.idx -> generated answer

    # ---- Phase 1: run all test cases through ask() ----
    for case in test_cases:
        result = ask(case.question)
        route_ok = RouterAccuracyMetric._compare(
            case.expected_source, result["route"]
        )
        case_route_marks[case.idx] = route_ok
        case_contexts[case.idx] = result["contexts"]
        case_answers[case.idx] = result["answer"]

        if result["route"]["source"] == "none":
            none_idxs.add(case.idx)
        else:
            deep_eval_cases.append(
                LLMTestCase(
                    name=str(case.idx),
                    input=case.question,
                    actual_output=result["answer"],
                    expected_output=case.ground_truth,
                    retrieval_context=result["contexts"],
                    additional_metadata={
                        "expected_source": case.expected_source,
                        "route": result["route"],
                    },
                )
            )

    # ---- Phase 2: batch DeepEval scoring ----
    json_path: str | None = None
    if deep_eval_cases:
        try:
            evaluate(
                test_cases=deep_eval_cases,
                metrics=ALL_METRICS,
                display_config=DisplayConfig(
                    print_results=False,
                    show_indicator=True,
                    results_folder="./data",
                ),
            )
        except Exception as exc:
            print(f"\n[WARNING] DeepEval evaluate() failed: {exc}")
            print("  Continuing with report (scores will be N/A).")
        else:
            # Find the JSON that was just written
            files = sorted(glob.glob(os.path.join("./data", "test_run_*.json")))
            if files:
                json_path = files[-1]  # latest
            else:
                print("\n[WARNING] No JSON saved — report will show N/A.")

    # ---- Phase 3: report (reads from JSON for guaranteed consistency) ----
    _print_report(test_cases, json_path, none_idxs, case_route_marks,
                  case_contexts, case_answers)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(
    test_cases: list[TestCase],
    json_path: str | None,
    none_idxs: set[int],
    case_route_marks: dict[int, bool],
    case_contexts: dict[int, list[str]],
    case_answers: dict[int, str],
) -> None:
    import json as _json

    total = len(test_cases)
    SEP = "=" * 70

    # Build scored lookup from JSON (not from eval_result — guaranteed match)
    scored_lookup: dict[int, dict[str, float]] = {}
    metric_names: list[str] = []

    if json_path is None:
        print(f"\n[WARNING] No test run JSON available — scores are N/A.")

    if json_path:
        with open(json_path) as f:
            data = _json.load(f)

        for tc_data in data.get("testCases", []):
            name = tc_data.get("name", "")
            if name.isdigit():
                case_idx = int(name)
                scored_lookup[case_idx] = {}
                for md in tc_data.get("metricsData", []):
                    score = md.get("score")
                    if score is not None:
                        scored_lookup[case_idx][md["name"]] = score

        # Derive metric names from first scored case
        if scored_lookup:
            first_scores = next(iter(scored_lookup.values()))
            metric_names = list(first_scores.keys())

    # ---- 1. Aggregate scores ----
    print(f"\n{SEP}")
    print("  1. AGGREGATE DEEPEVAL SCORES")
    print(SEP)
    for name in metric_names:
        vals = [scores.get(name) for scores in scored_lookup.values()]
        vals = [v for v in vals if v is not None]
        mean = sum(vals) / len(vals) if vals else float("nan")
        label = _METRIC_LABELS.get(name, name[:2])
        print(f"  {label:>4}  {name:<32}  {mean:.2f}")

    # Router accuracy computed from all 40 cases (not just scored subset)
    route_correct = sum(1 for v in case_route_marks.values() if v)
    route_mean = route_correct / total if total else 0.0
    print(f"  {'RT':>4}  {'Router Accuracy':<32}  {route_mean:.2f}")
    print()

    # ---- 2. Router accuracy (aggregate from all 40 cases, + misroute list) ----
    route_correct = sum(1 for v in case_route_marks.values() if v)
    print(SEP)
    print("  2. ROUTER ACCURACY")
    print(SEP)
    print(f"  {route_correct}/{total}  ({route_correct / total:.1%})")
    misrouted = [
        c for c in test_cases
        if not case_route_marks.get(c.idx, False)
    ]
    if misrouted:
        print()
        for case in misrouted:
            # We don't store routes separately anymore; print expected only
            print(f"  no #{case.idx}  expected={case.expected_source}")
            print(f"    Q: {case.question[:100]}...")
    print()

    # ---- 3. Per-question table ----
    print(SEP)
    print("  3. PER-QUESTION RESULTS")
    print(SEP)
    short_names = [
        _METRIC_LABELS.get(n, n[:2]) for n in metric_names
    ] if metric_names else ["F", "CR", "CP", "GE", "RT"]

    header = (
        f"  {'#':>3} {'Question':<48}"
        + "".join(f" {s:>6}" for s in short_names)
        + f" {'Route':>6}"
    )
    print(header)
    print(f"  {'-' * (72 + len(short_names) * 7)}")

    for case in test_cases:
        q = (
            case.question[:45] + "..."
            if len(case.question) > 48
            else case.question.ljust(48)
        )
        idx = case.idx
        route_mark = "yes" if case_route_marks.get(idx, False) else "no"

        if idx in none_idxs:
            blanks = "".join(" " * 7 for _ in short_names)
            print(f"  {idx:>3} {q}{blanks} NONE   {route_mark:>6}")
        elif idx in scored_lookup:
            scores = scored_lookup[idx]
            score_strs = ""
            for n in metric_names:
                v = scores.get(n)
                if v is not None:
                    score_strs += f" {v:>6.2f}"
                else:
                    score_strs += f" {'N/A':>6}"
            print(f"  {idx:>3} {q}{score_strs} {route_mark:>6}")
        else:
            blanks = "".join(" " * 7 for _ in short_names)
            print(f"  {idx:>3} {q}{blanks} MISS   {route_mark:>6}")

    # ---- 4. Context dump for retrieval failures (CR = 0.00) ----
    _CR_ZERO = "Contextual Recall"
    if scored_lookup:
        cr_zero_idxs = {
            idx for idx, scores in scored_lookup.items()
            if scores.get(_CR_ZERO, 1.0) == 0.0
        }
        if cr_zero_idxs:
            print()
            print(SEP)
            print("  4. RETRIEVED CONTEXTS (CR = 0.00 — retrieval failures)")
            print(SEP)
            for idx in sorted(cr_zero_idxs):
                case = next((c for c in test_cases if c.idx == idx), None)
                if case is None:
                    continue
                contexts = case_contexts.get(idx, [])
                answer = case_answers.get(idx, "")
                print(f"\n  --- #{idx}  Q: {case.question[:100]}")
                print(f"  Route: {case.expected_source}")
                print(f"  Answer: {answer[:120]}")
                print(f"  Contexts retrieved: {len(contexts)}")
                if not contexts:
                    print(f"  (empty — all chunks above 0.7 cosine distance threshold)")
                for i, c in enumerate(contexts):
                    truncated = c[:200].replace('\n', ' ')
                    print(f"  [{i}] ({len(c)} chars) {truncated}...")
    print()


# ---------------------------------------------------------------------------
# JSON result reader
# ---------------------------------------------------------------------------


def read_results(folder: str = "./data") -> dict | None:
    """Read the most recent test run JSON from *folder* and print a summary.

    Returns the parsed JSON dict, or ``None`` if no results found.
    """
    import glob
    import json

    files = sorted(glob.glob(os.path.join(folder, "test_run_*.json")))
    if not files:
        print(f"No test run JSON files found in {folder}")
        return None

    path = files[-1]  # latest run
    with open(path) as f:
        data = json.load(f)

    test_results = data.get("testCases", [])
    if not test_results:
        print(f"No test cases in {path}")
        return data

    # Collect per-metric scores
    metric_totals: dict[str, list[float]] = {}
    for tr in test_results:
        for md in tr.get("metricsData", []):
            name = md.get("name", "unknown")
            score = md.get("score")
            if score is not None:
                metric_totals.setdefault(name, []).append(score)

    print(f"\nResults loaded from {os.path.basename(path)}")
    print(f"  Test cases: {len(test_results)}")
    for name, scores in metric_totals.items():
        avg = sum(scores) / len(scores)
        print(f"  {name:<35}  {avg:.2f}  (n={len(scores)})")
    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
