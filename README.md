# 📚 RAG — Multi Source Information Retrieval

> A hybrid Retrieval-Augmented Generation (RAG) project built to learn and implement context injection using the DeepSeek API.

The system answers complex queries about the UI GreenMetric Sustainable University Rankings by combining unstructured narrative guidelines with structured tabular appendices.

**v1.0** · Python 3.12.13 · [Live Demo](https://fortunius-rag-uigreenmetric.hf.space)

---

## 🛠️ Tech Stack

| Layer | Component |
|---|---|
| Language | Python 3.12.13 |
| Embedding | `BAAI/bge-m3` (SentenceTransformers, 1024-dim) |
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
| `src/embedder.py` | Loads BGE-M3, encodes text into vectors, persists to ChromaDB |
| `src/retriever.py` | Single-query retrieval + multi-query RRF merge; dispatches by source (pdf/csv/both) and query type (lookup/aggregate) |
| `src/router.py` | LLM-based query classifier — routes to source (PDF/CSV/Both/None) and query type (lookup/aggregate); generates paraphrase variants for RAG Fusion |
| `src/generator.py` | Formats context + calls DeepSeek to produce answers; flags low-confidence results |
| `src/pipeline.py` | Orchestrator — wires route → paraphrase → multi-query RRF → (opt-in reranker) → generate |
| `src/budget.py` | Token budget tracking with HF Datasets persistence; guards against API overspend |
| `src/conversation.py` | Logs user prompts + responses to HF Datasets for quality monitoring |
| `src/reranker.py` | BGE V2-M3 cross-encoder reranker (enabled by default) |
| `src/evaluate.py` | Pre-computes routes + paraphrases, runs pipeline, DeepEval batch scorer, 5-section report with context debugging and per-case timing |
| `build_collection.py` | Chunks all 7 sources, prints sanity check (317 expected), builds ChromaDB collection |
| `app.py` | Gradio chat UI |

---

## 🛠️ Dependencies

Key libraries beyond the standard Python data stack:

| Package | Version | Purpose |
|---|---|---|
| `chromadb` | 1.5.9 | Vector database |
| `sentence-transformers` | 5.5.1 | Embedding model (BGE-M3) |
| `openai` | 2.38.0 | DeepSeek API client |
| `gradio` | 6.14.0 | Web UI |
| `deepeval` | 4.0.4 | Evaluation metrics |
| `huggingface-hub` | ≥0.20 | HF Datasets storage for budget + conversation logs |
| `transformers` | 4.57.6 | LLM model loading |
| `FlagEmbedding` | 1.4.0 | BGE reranker + BGE-M3 embeddings |

---

## ⚙️ Pipeline Architecture

1. **Ingestion:** Markdown → Heading-Level Chunking, CSV → Group-Based Chunking (118 question groups, 6 green building categories, 6 smart building fields, 30 coordinator countries, 7 category weights, 3 emission scopes) → Embed with BGE-M3 → Store in ChromaDB.
2. **Retrieval & Generation:** User Query → Budget Guard → Router (LLM) → Paraphrase (3 variants via DeepSeek) → Multi-query ChromaDB search (top-k=10 each) → RRF (k=60) → top 7 chunks → (reranker, enabled by default) → Context Concatenation → DeepSeek LLM Generation.

---

## 📊 Evaluation (DeepEval)

| Metric | v0.5 (MiniLM) | v0.6 (BGE-M3) | v1.0 (BGE-M3 + BGE) |
|---|---|---|---|
| Faithfulness | 0.91 | 0.93 | **0.96** |
| Contextual Recall | 0.74 | 0.81 | **0.83** |
| Contextual Precision (NDCG@K) | 0.45 | 0.56 | **0.72** |
| G-Eval Correctness | 0.43 | 0.51 | **0.59** |
| Router Accuracy | 80.0% | 77.5% | **91.5%** |

*47 test cases. LLM-as-judge ±0.05-0.08 variance. See `test_cases/RERANKER_REPORT_V2.md` for full benchmark.*

---

## 🧠 Design Decisions

| Decision | Reason |
|---|---|
| **Cosine similarity** | Matches the training metric of the embedding model |
| **BGE-M3 over Qwen3** | Better CP on structured CSV + markdown data (0.60 solo vs 0.44). Native dense+sparse training signal. No instruction prompt needed. Qwen3 evaluated and reverted. |
| **RAG Fusion (paraphrase ×3 + RRF)** | Resolves vocabulary mismatches that cosine search alone misses |
| **Question-grouped CSV chunks** | Prevents partial/orphaned indicators — the LLM always sees a complete criterion |
| **Formula injection in chunks** | Embedding formulas directly into chunk text reduces hallucination on calculation questions |
| **No cosine distance threshold** | Removed 0.5 threshold — was discarding relevant chunks; RRF handles quality ordering |
| **Single ChromaDB collection** | At 317 chunks, per-source collections add complexity with no performance gain |
| **No conversation history** | Degrades router accuracy — few-shot training uses single queries, and prior-turn vocabulary pulls the router toward stale sources |
| **BGE reranker enabled** | Evaluated 5 rerankers across 2 embedders. BGE V2-M3 + BGE-M3 embedder gives best CP (0.72) and CR (0.83). Qwen3+reranker hits 0.61 CP. Reranker enabled by default. |

---

## 🗺️ v0.5 → v1.0 Roadmap

- [x] Embedding model upgrade to **BGE-M3** (re-adopted after Qwen3 evaluation)
- [x] Rebuild ChromaDB collection (317 chunks, 1024-dim)
- [x] Embedding model upgrade to **Qwen3-Embedding-0.6B** (evaluated, CP inferior, reverted to BGE-M3)
- [x] Implement **RAG Fusion** (paraphrase + multi-query RRF)
- [x] Evaluate 5 rerankers — BGE V2-M3 adopted, enabled by default
- [x] Budget management for API spending
- [x] Deploy on HuggingFace Spaces (`fortunius/rag-uigreenmetric`)
- [x] Aggregate query optimization (metadata-driven stats, zero LLM, token reduction 15K→2K)
- [x] Router tuned to 91-94% with 26 few-shot examples — LLM variance (±5-8%), some queries span both sources
- [x] ~~Dense + sparse hybrid retrieval using BGE-M3 flag embeddings~~ | Tested, result in table below:

| Metric | Dense | Hybrid | Δ |
|---|---|---|---|
| Faithfulness | 0.96 | 0.92 | -0.04 |
| Contextual Recall | 0.83 | 0.84 | +0.01 |
| Contextual Precision | 0.72 | 0.71 | -0.01 |
| G-Eval | 0.59 | 0.51 | -0.08 |
| Router Accuracy | 91.5% | 89.4% | -2.1% |
| Avg latency | 10,188ms | 12,444ms | +2,256ms |

Sparse added 2.2s latency with no CP gain and degraded G-Eval. Not worth the cost.

---

## 🗺️ v1.5 Roadmap

- [ ] Show retrieved context (collapsible view of chunks used per answer)
- [ ] Source citation (display which data source answered the query)
- [ ] Token cost per query (exact usage and estimated cost under each answer)
- [ ] Pipeline timing (response generation time displayed)
- [ ] Route badge color-coding (green/blue/purple/gray for pdf/csv/both/none)
- [ ] Markdown rendering (properly formatted lists, tables, bold text)
- [ ] Copy answer button (clipboard copy on each response)
- [ ] Welcome + example chips (clickable sample questions on empty chat)
- [ ] Feedback thumbs (±1 per answer, logged for future evaluation)
- [ ] Dark mode toggle (OS preference detection + manual toggle)

## 🚀 v2.0 Roadmap

- [ ] FastAPI backend (`POST /ask`, `GET /budget`, `GET /health`)
- [ ] Streaming generator (DeepSeek `stream=True` → `StreamingResponse`)
- [ ] Conversation history (generator-only, router stays single-query)
- [ ] Dockerize backend (`Dockerfile` + `docker-compose.yml`)
- [ ] Svelte chat UI with Vercel AI SDK (replaces Gradio)
- [ ] Strip Gradio (`app.py`, `gradio` from requirements)

---

## ⚠️ Known Limitations

- **Router accuracy ~91-94%:** Improved with few-shot tuning but 1-4 cases still misrouted per run due to LLM variance (±5-8%). Some queries genuinely span both PDF and CSV sources — neither route is wrong, just incomplete.
- **CP bottleneck (0.72):** Contextual Precision remains the weakest metric, improving from 0.60 (BGE-M3 solo) to 0.72 (BGE-M3 + reranker + aggregate stats). Sparse hybrid retrieval tested and rejected. Further gains likely require embedder fine-tuning on domain-specific data.
- **G-Eval language sensitivity:** Scoring dips when the answer and ground truth differ in language (EN ↔ ID) despite being semantically equivalent.
- **RAG Fusion latency:** Paraphrase LLM call + 4× embeddings adds ~1-2s per query vs single-query retrieval.

---

