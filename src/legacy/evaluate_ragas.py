# """
# Evaluation pipeline for the UI GreenMetric RAG system.
# 
# This script evaluates the performance of the RAG system using a set of predefined metrics. It loads the evaluation dataset, 
# runs the RAG system on each sample, and computes the metrics to assess the quality of the generated responses.
# """
# 
# import os
# import sys
# import typing as t
# import langchain_google_vertexai
# sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai
# 
# import pandas as pd
# from openai import OpenAI
# from dotenv import load_dotenv
# 
# from ragas.llms import llm_factory
# from ragas import evaluate, EvaluationDataset, SingleTurnSample
# from ragas.metrics import Faithfulness, ContextRecall, ResponseRelevancy, AnswerCorrectness
# from ragas.embeddings.base import BaseRagasEmbedding
# 
# #----------------------------------------------------------------------------
# # Environment and LLM setup
# #----------------------------------------------------------------------------
# load_dotenv()
# 
# LLM_CLIENT = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# 
# RAGAS_LLM = llm_factory(
#     "deepseek-v4-flash",
#     provider="openai",
#     client=LLM_CLIENT,
#     extra_body={"thinking": {"type": "disabled"}},
#     max_tokens=8192,
# )
# 
# #----------------------------------------------------------------------------
# # Custom embedding wrapper
# #----------------------------------------------------------------------------
# 
# import asyncio
# from src.embedder import EMBED_MODEL
# 
# class _STEmbedding(BaseRagasEmbedding):
#     def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
#         return EMBED_MODEL.encode(text, show_progress_bar=False).tolist()
#     async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
#         return await asyncio.to_thread(self.embed_text, text, **kwargs)
#     def embed_query(self, text: str) -> t.List[float]:
#         return self.embed_text(text)
#     def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
#         return self.embed_texts(texts)
# 
# RAGAS_EMBEDDINGS = _STEmbedding()
# 
# #----------------------------------------------------------------------------
# # Load test cases
# #----------------------------------------------------------------------------
# 
# from dataclasses import dataclass
# 
# @dataclass
# class TestCase:
#     idx: int
#     question: str
#     ground_truth: str
#     expected_source: str
#     notes: str = ""
# 
# def load_test_cases(path: str = "test_cases/test_cases.xlsx") -> t.List[TestCase]:
#     df = pd.read_excel(path)
#     cases: list[TestCase] = []
#     for i, row in df.iterrows():
#         cases.append(
#             TestCase(
#                 idx=i,
#                 question=row["question"],
#                 ground_truth=row["ground_truth"],
#                 expected_source=row["source"],
#                 notes=str(row.get("notes", "")) if pd.notna(row.get("notes", "")) else "",
#             )
#         )
#     return cases
# 
# #----------------------------------------------------------------------------
# # Route comparison
# #----------------------------------------------------------------------------
# 
# def _route_matches(expected: str, route: dict) -> bool:
#     if expected == "none":
#         return route["source"] == "none"
#     if expected == "pdf":
#         return route["source"] == "pdf"
#     if expected.startswith("both_pdf_"):
#         csv = expected[len("both_pdf_"):]
#         return route["source"] == "both" and route.get("csv_source") == csv
#     if expected.startswith("csv_"):
#         return route["source"] == "csv" and route.get("csv_source") == expected
#     return False
# 
# #----------------------------------------------------------------------------
# # Refusal detection
# #----------------------------------------------------------------------------
# 
# _REFUSAL_PATTERNS = [
#     "i don't have the required information",
#     "the information regarding your question is not available",
#     "the guideline document does not",
#     "the document does not",
#     "not available in our document",
#     "not within the scope",
#     "sorry, i don't know",
#     "cannot answer",
# ]
# 
# def _is_refusal(answer: str) -> bool:
#     lowered = answer.lower()
#     for pat in _REFUSAL_PATTERNS:
#         if pat in lowered:
#             return True
#     return False
# 
# #----------------------------------------------------------------------------
# # Run evaluation
# #----------------------------------------------------------------------------
# 
# from src.pipeline import ask
# 
# def run_evaluation(path: str = "test_cases/test_cases.xlsx") -> None:
#     test_cases = load_test_cases(path=path)
#     samples: list[SingleTurnSample] = []
#     sample_idxs: list[int] = []
#     none_results: list[tuple[TestCase, str, bool]] = []
#     route_correct = 0
#     case_routes: list[tuple[TestCase, bool, dict]] = []
# 
#     for case in test_cases:
#         result = ask(case.question)
#         print("----------" * 3)
#         print(f"Test case {case.idx}: ")
#         print(f"Question={case.question}")
#         print(f"Expected source={case.expected_source}")
#         print("Result:")
#         print(f"  → route={result['route']}")
#         print(f"  → answer={result['answer']}")
#         print("----------" * 3)
#         print()
# 
#         route_ok = _route_matches(case.expected_source, result["route"])
#         if route_ok:
#             route_correct += 1
#         case_routes.append((case, route_ok, result["route"]))
# 
#         if result["route"]["source"] == "none":
#             none_results.append(
#                 (case, result["answer"], _is_refusal(result["answer"]))
#             )
#         else:
#             samples.append(
#                 SingleTurnSample(
#                     user_input=case.question,
#                     response=result["answer"],
#                     retrieved_contexts=result["contexts"],
#                     reference=case.ground_truth,
#                 )
#             )
#             sample_idxs.append(case.idx)
# 
#     dataset = EvaluationDataset(samples=samples)
#     scored = evaluate(
#         dataset=dataset,
#         metrics=[
#             Faithfulness(),
#             ContextRecall(),
#             ResponseRelevancy(),
#             AnswerCorrectness(),
#         ],
#         llm=RAGAS_LLM,
#         embeddings=RAGAS_EMBEDDINGS,
#     )
# 
#     df = scored.to_pandas()
#     df["_case_idx"] = sample_idxs
# 
#     _print_report(test_cases, df, case_routes, none_results)
# 
# 
# def _print_report(
#     test_cases: list[TestCase],
#     df: pd.DataFrame,
#     case_routes: list[tuple[TestCase, bool, dict]],
#     none_results: list[tuple[TestCase, str, bool]],
# ) -> None:
#     metric_cols = ["faithfulness", "context_recall", "answer_relevancy", "answer_correctness"]
#     labels = {"faithfulness": "F", "context_recall": "CR",
#               "answer_relevancy": "RR", "answer_correctness": "AC"}
#     total = len(test_cases)
#     route_correct = sum(1 for _, ok, _ in case_routes if ok)
#     none_idxs = {n[0].idx for n in none_results}
#     SEP = "=" * 70
# 
#     print(f"\n{SEP}")
#     print("  1. AGGREGATE RAGAS SCORES")
#     print(SEP)
#     for col in metric_cols:
#         vals = df[col].dropna()
#         mean = vals.mean() if not vals.empty else float("nan")
#         print(f"  {labels[col]:>4}  {col:<25}  {mean:.4f}")
#     print()
# 
#     print(SEP)
#     print("  2. PER-SOURCE BREAKDOWN")
#     print(SEP)
#     df["_source"] = df["_case_idx"].map({c.idx: c.expected_source for c in test_cases})
#     header = f"  {'Source':<30} {'#':>3}"
#     for col in metric_cols:
#         header += f" {labels[col]:>8}"
#     print(header)
#     print(f"  {'-'*30} {'-'*3} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
#     for src in sorted(df["_source"].unique()):
#         sub = df[df["_source"] == src]
#         row = f"  {src:<30} {len(sub):>3}"
#         for col in metric_cols:
#             vals = sub[col].dropna()
#             row += f" {vals.mean():>8.4f}" if not vals.empty else "      N/A"
#         print(row)
#     print()
# 
#     print(SEP)
#     print("  3. ROUTER ACCURACY")
#     print(SEP)
#     print(f"  {route_correct}/{total}  ({route_correct / total:.1%})")
#     misrouted = [(c, r) for c, ok, r in case_routes if not ok]
#     if misrouted:
#         print()
#         for case, route in misrouted:
#             r_src = route["source"]
#             r_csv = route.get("csv_source")
#             r_str = r_src + (f"/{r_csv}" if r_csv else "")
#             print(f"  ✗ #{case.idx}  expected={case.expected_source}  got={r_str}")
#             print(f"    Q: {case.question[:100]}...")
#     print()
# 
#     print(SEP)
#     print("  4. WEAKEST 5 (by avg RAGAS score)")
#     print(SEP)
#     cols = metric_cols + ["_case_idx"]
#     scored = df[cols].dropna(how="all", subset=metric_cols).copy()
#     scored["avg"] = scored[metric_cols].mean(axis=1)
#     weakest = scored.nsmallest(5, "avg")
#     case_lookup = {c.idx: c for c in test_cases}
#     for i, (_, row) in enumerate(weakest.iterrows()):
#         case = case_lookup[int(row["_case_idx"])]
#         print(f"\n  #{i + 1}  avg={row['avg']:.4f}  {case.question[:90]}...")
#         for col in metric_cols:
#             if pd.notna(row[col]):
#                 print(f"       {labels[col]:>4}  {col:<25}  {row[col]:.4f}")
#     print()
# 
#     print(SEP)
#     print("  5. PER-QUESTION RESULTS")
#     print(SEP)
#     route_lookup = {c.idx: (ok, r) for c, ok, r in case_routes}
#     scored_lookup = {}
#     for _, row in df.iterrows():
#         scored_lookup[int(row["_case_idx"])] = row
#     h = f"  {'#':>3} {'Question':<48} {'F':>6} {'CR':>6} {'RR':>6} {'AC':>6} {'Route':>6}"
#     print(h)
#     print(f"  {'-' * 86}")
#     for case in test_cases:
#         q = case.question[:45] + "..." if len(case.question) > 48 else case.question.ljust(48)
#         idx = case.idx
#         r_ok, _ = route_lookup.get(idx, (False, {}))
#         route_mark = "✓" if r_ok else "✗"
# 
#         if idx in none_idxs:
#             print(f"  {idx:>3} {q} {'NONE':>6} {'':>6} {'':>6} {'':>6} {route_mark:>6}")
#         elif idx in scored_lookup:
#             row = scored_lookup[idx]
#             fv = f"{row['faithfulness']:.2f}" if pd.notna(row.get('faithfulness')) else "  N/A"
#             cv = f"{row['context_recall']:.2f}" if pd.notna(row.get('context_recall')) else "  N/A"
#             rv = f"{row['answer_relevancy']:.2f}" if pd.notna(row.get('answer_relevancy')) else "  N/A"
#             av = f"{row['answer_correctness']:.2f}" if pd.notna(row.get('answer_correctness')) else "  N/A"
#             print(f"  {idx:>3} {q} {fv:>6} {cv:>6} {rv:>6} {av:>6} {route_mark:>6}")
#     print()
