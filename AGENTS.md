# AGENTS.md

## Commands
- `python app.py` — launch Gradio UI
- `python3 -c "from src.evaluate import run_evaluation; run_evaluation('test_cases/test_cases.xlsx')"` — full 47-case eval
- `python3 -c "from src.evaluate import run_evaluation; run_evaluation('test_cases/test_cases_5.xlsx')"` — 5-case quick eval
- `RAG_BUDGET_TOKENS=999999999` — prefix before evals to bypass budget
- `python build_collection.py` — rebuild ChromaDB (317 chunks)
- `RAG_RERANK=0` — disable reranker (enabled by default)

## Conventions
- No new dependencies without discussion
- Evaluation results go in `test_cases/*.md`, not README
- ChromaDB pre-built and committed — rebuild only on embedder/chunker change
- Source code in `src/`, data in `rag_data/`, tests in `test_cases/`
- All secrets via `os.getenv()`, never hardcoded

## Don't
- **Never run evaluation without explicit permission** — a 47-case eval costs ~250K+ tokens ($0.10+)
- **Never commit without explicit permission** — the user prefers manual review before commits
- Don't modify the router few-shot examples without running full eval after
- Don't add code to `src/summarizer.py` (deactivated, keep as reference)
- Don't hardcode API keys — use `os.getenv()`
- Don't modify `.gitignore` entries for `src/legacy/` and `rag_data/old_pdf/` (kept as reference)
