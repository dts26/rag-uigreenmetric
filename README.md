# 📚 RAG — Multi Source Information Retrieval

> A hybrid RAG system that answers complex queries about the UI GreenMetric Sustainable University Rankings, combining unstructured narrative guidelines with structured tabular appendices.

**v0.5.1** · Python 3.12.13

---

## 🛠️ Tech Stack

| Layer | Component |
|---|---|
| Language | Python 3.12.13 |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers, 384-dim) |
| Vector DB | ChromaDB (cosine distance, top-k=5) |
| Data | `pandas`, `openpyxl` |
| LLM (pipeline) | DeepSeek API via OpenAI SDK (`v4-pro` generation, `v4-flash` routing) |
| LLM (evaluation) | `deepseek-v4-flash` with thinking disabled |
| UI | Gradio |
| Evaluation | DeepEval 4.0.4 |

---

## 🗂️ Data Sources

The UI GreenMetric guidelines document is split into **7 files** — 1 narrative document and 6 structured tables. This separation preserves each piece's natural format and lets the retriever target the right structure per query.

| File | Format | Content |
|---|---|---|
| `guidelines_markdown.md` | Markdown | Full narrative: methodology, categories, indicators, evidence rules, coordinator info |
| `appendix1_questionnairemasterandscoring.csv` | CSV (question-grouped) | 118 indicators across 7 categories with answer options and calculated scores |
| `appendix2_listofgreenbuildingelements.csv` | CSV (category-grouped) | Green building elements for existing buildings and new construction |
| `appendix3_listanddescriptionofsmartbuildingrequirements.csv` | CSV (field-code-grouped) | Smart building requirements (Automation, Safety, Energy, Water, Indoor, Lighting) |
| `table1_nationalcoordinators.csv` | CSV (country-grouped) | 35 national coordinator universities across 30 countries |
| `table2_categoriesusedandweighting.csv` | CSV (category-grouped) | 7 categories with their percentage weights |
| `table4_greenhousegasemissionsources.csv` | CSV (scope-grouped) | Emission sources classified by Scope 1, 2, and 3 |

**Why split tables out of the PDF?** Tables extracted during PDF-to-markdown conversion lose their row-column structure. By storing them as separate CSVs, the retriever can fetch data by natural grouping keys (question number, category, country) instead of parsing broken table text from paragraphs.

---

## 🧱 Source Code

| File | Role |
|---|---|
| `src/chunker.py` | Splits markdown (heading-level) and CSV tables (grouped by column) into embeddable chunks |
| `src/embedder.py` | Loads the embedding model, encodes text into vectors, persists to ChromaDB |
| `src/retriever.py` | Cosine similarity search with 0.7 threshold guardrail; full-source fetch for aggregate queries |
| `src/router.py` | LLM-based query classifier — routes to source (PDF/CSV/Both/None) and query type (lookup/aggregate) |
| `src/generator.py` | Formats context + calls DeepSeek to produce answers; flags low-confidence results |
| `src/pipeline.py` | Orchestrator — wires router → retriever → generator into a single `ask()` call |
| `src/evaluate.py` | Test case loader + DeepEval batch scorer + 5-section report with context debugging |
| `app.py` | Gradio chat UI |

---

## ⚙️ Pipeline Architecture

1. **Unstructured Data:** Markdown → Heading-Level Chunking → Embed → Store in ChromaDB.
2. **Structured Data (CSV):** Question-Grouped Chunks (combining criteria, options, and injected formulas) → Embed → Store in ChromaDB.
3. **Retrieval & Generation:** User Query → Router (LLM) → Cosine Similarity Search → 0.7 Low Confidence Filter → Context Concatenation → DeepSeek LLM Generation.
4. **Router Agent:** Classifies queries by source (PDF, CSV, Both, None) and query type (lookup, aggregate) before retrieval.

---

## 📊 Evaluation (DeepEval, v0.5.1)

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | **0.91** | Whether the answer is factually supported by the retrieved context |
| Contextual Recall | **0.74** | Whether retrieved context contains all information needed to answer |
| Contextual Precision (NDCG@K) | **0.45** | Whether relevant chunks are ranked higher in results |
| G-Eval Correctness | **0.43** | How well the answer matches the ground truth (LLM-as-judge, 0–1) |
| Router Accuracy | **80.0%** | Whether the pipeline router correctly identifies the expected data source |

***LLM-as-judge metrics show run-to-run variance of ±0.05–0.08 even at temperature 0**

---

## 🧠 Design Decisions

| Decision | Reason |
|---|---|
| **Cosine similarity** | Matches the training metric of the embedding model |
| **Question-grouped CSV chunks** | Prevents partial/ orphaned indicators — the LLM always sees a complete criterion |
| **Formula injection in chunks** | Embedding formulas directly into chunk text reduces hallucination on calculation questions |
| **0.7 cosine distance threshold** | Empirically calibrated guardrail on this dataset |
| **Top-K = 5** | Multiple chunks improve synthesis for cross-indicator questions |
| **Single ChromaDB collection** | At 317 chunks, per-source collections add complexity with no performance gain |
| **No conversation history** | Degrades router accuracy — few-shot training uses single queries, and prior-turn vocabulary pulls the router toward stale sources |

---

## 🗺️ v0.5 → v1.0 Roadmap

- Embedding model upgrade to **BGE-M3**
- Add **bge-reranker-v2-m3** for re-ranking
- Add **RAG Fusion** for multi-query retrieval
- Budget management for API spending

---

## ⚠️ Known Limitations

- **Router accuracy ~80%:** Misclassifies `both`-source queries as single-source, or `pdf` as `csv`. Likely due to under-trained few-shot examples for edge cases.
- **Aggregate queries use brute-force:** `_fetch_all` returns every chunk — clean but risks context window overflow.
- **Vocabulary mismatch:** "UI GM" ≠ "UI GreenMetric", "created" ≠ "initiated" — semantic gaps cause retrieval misses.
- **G-Eval language sensitivity:** Scoring dips when the answer and ground truth differ in language (EN ↔ ID) despite being semantically equivalent.
- **Contextual Precision = 0.45:** Relevant chunks are often ranked low — a re-ranker would help.

---
