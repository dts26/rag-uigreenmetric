# 📚 RAG — Multi Source Information Retrieval

> A hybrid Retrieval-Augmented Generation (RAG) project built to learn and implement context injection using the DeepSeek API.

The system answers complex queries about the UI GreenMetric Sustainable University Rankings by combining unstructured narrative guidelines with structured tabular appendices.

**v0.8.0** · Python 3.12.13

---

## 🛠️ Tech Stack

| Layer | Component |
|---|---|
| Language | Python 3.12.13 |
| Embedding | `Qwen/Qwen3-Embedding-0.6B` (SentenceTransformers, 1024-dim) |
| Retrieval | RAG Fusion: paraphrase ×3 + ChromaDB (cosine, top-k=10) + RRF (k=60), top-n=7 |
| Data | `pandas`, `openpyxl` |
| LLM (pipeline) | DeepSeek API via OpenAI SDK (`v4-pro` generation, `v4-flash` routing/paraphrase) |
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
| `src/embedder.py` | Loads Qwen3-Embedding (local or HF Inference API via `EMBED_BACKEND=hf_api`), encodes text into vectors, persists to ChromaDB |
| `src/retriever.py` | Single-query retrieval + multi-query RRF merge; dispatches by source (pdf/csv/both) and query type (lookup/aggregate) |
| `src/router.py` | LLM-based query classifier — routes to source (PDF/CSV/Both/None) and query type (lookup/aggregate); generates paraphrase variants for RAG Fusion |
| `src/generator.py` | Formats context + calls DeepSeek to produce answers; flags low-confidence results |
| `src/pipeline.py` | Orchestrator — wires route → paraphrase → multi-query RRF → (opt-in reranker) → generate |
| `src/budget.py` | Token budget tracking with HF Datasets persistence; guards against API overspend |
| `src/conversation.py` | Logs user prompts + responses to HF Datasets for quality monitoring |
| `src/reranker.py` | Optional Jina V3 cross-encoder reranker (not recommended — see reports) |
| `src/evaluate.py` | Pre-computes routes + paraphrases, runs pipeline, DeepEval batch scorer, 5-section report with context debugging and per-case timing |
| `build_collection.py` | Chunks all 7 sources, prints sanity check (317 expected), builds ChromaDB collection |
| `app.py` | Gradio chat UI |

---

## 🛠️ Dependencies

Key libraries beyond the standard Python data stack:

| Package | Version | Purpose |
|---|---|---|
| `chromadb` | 1.5.9 | Vector database |
| `sentence-transformers` | 5.5.1 | Embedding model (Qwen3) |
| `openai` | 2.38.0 | DeepSeek API client |
| `gradio` | 6.14.0 | Web UI |
| `deepeval` | 4.0.4 | Evaluation metrics |
| `huggingface-hub` | ≥0.20 | HF Datasets storage for budget + conversation logs |
| `transformers` | 4.57.6 | LLM model loading |

---

## ⚙️ Pipeline Architecture

1. **Ingestion:** Markdown → Heading-Level Chunking, CSV → Group-Based Chunking (118 question groups, 6 green building categories, 6 smart building fields, 30 coordinator countries, 7 category weights, 3 emission scopes) → Embed with Qwen3 (no instruction) → Store in ChromaDB.
2. **Retrieval & Generation:** User Query → Budget Guard → Router (LLM) → Paraphrase (3 variants via DeepSeek) → Multi-query ChromaDB search (top-k=10 each) → RRF (k=60) → top 7 chunks → Context Concatenation → DeepSeek LLM Generation.
3. **Query encoding:** Queries use a domain-specific instruction prompt (`Instruct: Given a question about UI GreenMetric university sustainability rankings, retrieve relevant guideline documents and indicator data`). Documents are encoded raw.

---

## 📊 Evaluation (DeepEval)

| Metric | v0.5 (MiniLM) | v0.6 (BGE-M3) | v0.8 (Qwen3, RAG Fusion) |
|---|---|---|---|
| Faithfulness | 0.91 | 0.93 | **0.95** |
| Contextual Recall | 0.74 | 0.81 | **0.83** |
| Contextual Precision (NDCG@K) | 0.45 | 0.56 | **0.50** |
| G-Eval Correctness | 0.43 | 0.51 | **0.59** |
| Router Accuracy | 80.0% | 77.5% | **89.4%** |

*47 test cases (v0.8 includes 7 synthetic). Scores not directly comparable to v0.5/v0.6 (40 cases) — the expanded test set is harder. LLM-as-judge metrics: ±0.05–0.08 variance. See `test_cases/RF_RERANKER_REPORT.md` for full benchmark.*

---

## 🧠 Design Decisions

| Decision | Reason |
|---|---|
| **Cosine similarity** | Matches the training metric of the embedding model |
| **Qwen3-Embedding over BGE-M3** | +10 points on MTEB retrieval (64.64 vs 54.60), instruction-aware encoding, longer 32K context |
| **Custom query instruction** | Domain-specific prompt improves retrieval accuracy vs generic "web search" default |
| **RAG Fusion (paraphrase ×3 + RRF)** | Resolves vocabulary mismatches that cosine search alone misses |
| **Question-grouped CSV chunks** | Prevents partial/orphaned indicators — the LLM always sees a complete criterion |
| **Formula injection in chunks** | Embedding formulas directly into chunk text reduces hallucination on calculation questions |
| **No cosine distance threshold** | Removed 0.5 threshold — was discarding relevant chunks; RRF handles quality ordering |
| **Single ChromaDB collection** | At 317 chunks, per-source collections add complexity with no performance gain |
| **No conversation history** | Degrades router accuracy — few-shot training uses single queries, and prior-turn vocabulary pulls the router toward stale sources |
| **No reranker** | Evaluated 5 rerankers (BGE, GTE, Nemotron, Qwen3, Jina). None add value on top of Qwen3 + RAG Fusion. Jina harms CR (-0.07) and GE (-0.06). GTE causes G-Eval collapse. Rerankers disabled by default. |

---

## 🗺️ v0.5 → v1.0 Roadmap

- [x] Embedding model upgrade to **BGE-M3** (superseded by Qwen3)
- [x] Rebuild ChromaDB collection (317 chunks, 1024-dim)
- [x] Embedding model upgrade to **Qwen3-Embedding-0.6B**
- [x] Implement **RAG Fusion** (paraphrase + multi-query RRF)
- [x] Evaluate 5 rerankers (BGE, GTE, Nemotron, Qwen3, Jina) — none recommended, disabled by default
- [x] Budget management for API spending
- [x] Deploy on HuggingFace Spaces (`fortunius/rag-uigreenmetric`)
- [ ] Router tuned to 89.4% with few-shot examples — 3-5 cases still misrouted; add targeted examples

---

## ⚠️ Known Limitations

- **Router accuracy ~89%:** Improved with few-shot tuning but 3-5 cases still misrouted (mostly pdf→csv). Adding more targeted examples would help.
- **Aggregate queries use brute-force (planned for v1.0):** `_fetch_all` returns all 118 chunks for queries like "Which category has the most questions?" — correct but wastes ~15-20K tokens. Two fix paths: a pre-built summary (correct, fast, but manual to maintain) or directed LLM summarization instructing the model to capture key details (category counts, min/max scores, coordinator names, emission scopes) for the downstream generator.
- **CP bottleneck (~0.50):** Contextual Precision is the hardest metric to move. Embedder upgrade (BGE-M3 → Qwen3), RAG Fusion, and 5 rerankers all failed to raise it past ~0.55 on 40 cases. Improving this requires embedding model fine-tuning on domain-specific data.
- **G-Eval language sensitivity:** Scoring dips when the answer and ground truth differ in language (EN ↔ ID) despite being semantically equivalent.
- **RAG Fusion latency:** Paraphrase LLM call + 4× embeddings adds ~1-2s per query vs single-query retrieval.

---

