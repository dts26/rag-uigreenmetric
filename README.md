# 📚 RAG — Multi Source Information Retrieval

> A hybrid Retrieval-Augmented Generation (RAG) project built to learn and implement context injection using the DeepSeek API.

The system answers complex queries about the UI GreenMetric Sustainable University Rankings by combining unstructured narrative guidelines with structured tabular appendices.

**v0.6.0** · Python 3.12.13

---

## 🛠️ Tech Stack

| Layer | Component |
|---|---|
| Language | Python 3.12.13 |
| Embedding | `BAAI/bge-m3` (SentenceTransformers, 1024-dim) |
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
| `src/retriever.py` | Cosine similarity search with 0.5 threshold guardrail; full-source fetch for aggregate queries |
| `src/router.py` | LLM-based query classifier — routes to source (PDF/CSV/Both/None) and query type (lookup/aggregate) |
| `src/generator.py` | Formats context + calls DeepSeek to produce answers; flags low-confidence results |
| `src/pipeline.py` | Orchestrator — wires router → retriever → generator into a single `ask()` call |
| `src/evaluate.py` | Test case loader + DeepEval batch scorer + 5-section report with context debugging |
| `app.py` | Gradio chat UI |

---

## ⚙️ Pipeline Architecture

1. **Unstructured Data:** Markdown → Heading-Level Chunking → Embed → Store in ChromaDB.
2. **Structured Data (CSV):** Question-Grouped Chunks (combining criteria, options, and injected formulas) → Embed → Store in ChromaDB.
3. **Retrieval & Generation:** User Query → Router (LLM) → Cosine Similarity Search → 0.5 Low Confidence Filter → Context Concatenation → DeepSeek LLM Generation.
4. **Router Agent:** Classifies queries by source (PDF, CSV, Both, None) and query type (lookup, aggregate) before retrieval.

---

## 📊 Evaluation (DeepEval)

| Metric | v0.5 (MiniLM) | v0.6 (BGE-M3) | +Reranker |
|---|---|---|---|
| Faithfulness | 0.91 | **0.93** | — |
| Contextual Recall | 0.74 | **0.81** | — |
| Contextual Precision (NDCG@K) | 0.45 | **0.56** | — |
| G-Eval Correctness | 0.43 | **0.51** | — |
| Router Accuracy | 80.0% | **77.5%** | — |

*LLM-as-judge metrics: ±0.05–0.08 run-to-run variance at temperature 0.*

---

## 🧠 Design Decisions

| Decision | Reason |
|---|---|
| **Cosine similarity** | Matches the training metric of the embedding model |
| **Question-grouped CSV chunks** | Prevents partial/ orphaned indicators — the LLM always sees a complete criterion |
| **Formula injection in chunks** | Embedding formulas directly into chunk text reduces hallucination on calculation questions |
| ~~**0.7 cosine distance threshold**~~ | ~~Empirically calibrated guardrail on this dataset~~ — lowered to 0.5 in preparation for reranker |
| **Top-K = 5** | Multiple chunks improve synthesis for cross-indicator questions |
| **Single ChromaDB collection** | At 317 chunks, per-source collections add complexity with no performance gain |
| **No conversation history** | Degrades router accuracy — few-shot training uses single queries, and prior-turn vocabulary pulls the router toward stale sources |

---

## 🗺️ v0.5 → v1.0 Roadmap

- [x] Embedding model upgrade to **BGE-M3**
- [x] Rebuild ChromaDB collection (317 chunks, 1024-dim)
- [ ] Add **bge-reranker-v2-m3** for re-ranking
- [ ] Add **RAG Fusion** for multi-query retrieval
- [ ] Budget management for API spending
- [ ] Deploy on HuggingFace Spaces

---

## ⚠️ Known Limitations

- **Router accuracy ~80%:** Misclassifies `both`-source queries as single-source, or `pdf` as `csv`. Likely due to under-trained few-shot examples for edge cases.
- **Aggregate queries use brute-force:** `_fetch_all` returns every chunk — clean but risks context window overflow.
- **Vocabulary mismatch:** "UI GM" ≠ "UI GreenMetric", "created" ≠ "initiated" — semantic gaps cause retrieval misses.
- **G-Eval language sensitivity:** Scoring dips when the answer and ground truth differ in language (EN ↔ ID) despite being semantically equivalent.
- **Contextual Precision = 0.53:** Improved with BGE-M3 but still the weakest metric — a re-ranker should help.

---
