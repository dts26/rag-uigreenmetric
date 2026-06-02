# AGENTS.md

This file is loaded automatically by AI coding agents. Available skills from `~/.agents/skills/` (e.g., `hf-cli` for HuggingFace Hub operations) should be invoked when relevant.

## Commands
- `python app.py` — launch Gradio UI
- `python3 -c "from src.evaluate import run_evaluation; run_evaluation('test_cases/test_cases.xlsx')"` — full 47-case eval
- `python3 -c "from src.evaluate import run_evaluation; run_evaluation('test_cases/test_cases_5.xlsx')"` — 5-case quick eval
- `RAG_BUDGET_TOKENS=999999999` — prefix before evals to bypass budget
- `python build_collection.py` — rebuild ChromaDB (317 chunks)
- `RAG_RERANK=0` — disable reranker (enabled by default)

## File Layout

Every tracked file in the repo (excluding `.gitignore`-covered):

| File | Purpose |
|---|---|
| `AGENTS.md` | Rules for AI coding agents (this file) |
| `LICENSE` | Project license |
| `README.md` | Human-facing documentation, roadmap, evaluation scores |
| `app.py` | Gradio chat UI entry point |
| `build_collection.py` | Chunk all 7 sources → rebuild ChromaDB (317 chunks) |
| `requirements.txt` | Pinned Python dependencies |
| `rag_data/guidelines_markdown.md` | Full narrative: methodology, categories, indicators |
| `rag_data/appendix1_*` | 118 indicators with scores and options |
| `rag_data/appendix2_*` | Green building elements (existing + new) |
| `rag_data/appendix3_*` | Smart building requirements by field code |
| `rag_data/table1_*` | 35 coordinator universities across 30 countries |
| `rag_data/table2_*` | 7 categories with weight percentages |
| `rag_data/table4_*` | Emission sources by Scope 1/2/3 |
| `src/budget.py` | Token tracking with HF Dataset persistence |
| `src/chunker.py` | Markdown heading-level + CSV group-based chunking |
| `src/conversation.py` | Conversation logging to HF Dataset JSONL |
| `src/embedder.py` | BGE-M3 embedding + ChromaDB storage |
| `src/evaluate.py` | Test case loader + DeepEval scorer + 5-section report |
| `src/generator.py` | Context formatting + DeepSeek v4-pro generation |
| `src/legacy/evaluate_ragas.py` | Old RAGAS evaluation (reference, gitignored) |
| `src/legacy/ragas_result.txt` | Old RAGAS results (reference, gitignored) |
| `src/pipeline.py` | Orchestrator: route → paraphrase → RRF → rerank → generate |
| `src/reranker.py` | BGE V2-M3 cross-encoder (enabled by default) |
| `src/retriever.py` | ChromaDB search + multi-query RRF + aggregate stats |
| `src/router.py` | LLM query classifier (26 few-shots) + paraphrase generator |
| `src/summarizer.py` | LLM aggregator (deactivated, kept as reference, gitignored) |
| `test_cases/test_cases.xlsx` | 47 evaluation test cases |
| `test_cases/test_cases_5.xlsx` | 5-case quick eval subset |
| `test_cases/RF_RERANKER_REPORT.md` | RAG Fusion + reranker benchmark (v0.8) |
| `test_cases/RERANKER_REPORT_V2.md` | Full 8-config embedder + reranker benchmark (v1.0) |
| `chroma_db/` | Pre-built BGE-M3 ChromaDB (committed, 4.2 MB) |

## Conventions
- No new dependencies without discussion
- Evaluation results go in `test_cases/*.md`, not README
- ChromaDB pre-built and committed — rebuild only on embedder/chunker change
- Source code in `src/`, data in `rag_data/`, tests in `test_cases/`
- All secrets via `os.getenv()`, never hardcoded
- Ask for the human's confirmation before writing down any codes.

## Commit Messages
Format: `[TAG] short description`

| Tag | Use for |
|---|---|
| `[FIX]` | Bugs, typos, stale references, broken tables |
| `[ADD]` | New features, new files, new test cases |
| `[REMOVE]` | Dead code, unused deps, pruned files |
| `[DOCS]` | README, AGENTS.md, reports, docstrings |
| `[REFACTOR]` | Pipeline changes, embedder/reranker swaps, architecture |
| `[VERSION]` | Version bumps (v0.9, v1.0) |

Examples:
- `[FIX] budget guard now checks before every generate call`
- `[ADD] aggregate query optimization via metadata stats`
- `[REMOVE] hybrid search mentions from README`
- `[DOCS] update evaluation table with v1.0 scores`
- `[REFACTOR] switch back to BGE-M3 + BGE reranker`
- `[VERSION] v1.0 shipped`

## Don't
- **Never run evaluation without explicit permission** — a 47-case eval costs ~250K+ tokens ($0.10+)
- **Never commit without explicit permission** — before every commit, the human needs to review it first
- Don't modify the router few-shot examples without running full eval after
- Don't add code to `src/summarizer.py` (deactivated, keep as reference)
- Don't hardcode API keys — use `os.getenv()`
- Don't modify `.gitignore` entries for `src/legacy/` and `rag_data/old_pdf/` (kept as reference)
