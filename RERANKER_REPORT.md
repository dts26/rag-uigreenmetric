# Reranker Benchmark Report — UI GreenMetric RAG

**v0.6.0** · 2026-05-29 · 40 test cases · BGE-M3 embeddings + ChromaDB (top-k=20, threshold removed)

---

## Models Tested

| # | Model | Architecture | Params | Max Context | Languages | License | Integration |
|---|---|---|---|---|---|---|---|
| 1 | `BAAI/bge-reranker-v2-m3` | XLMRoberta cross-encoder | 568M | 512 | 75+ | Apache 2.0 | FlagEmbedding |
| 2 | `Alibaba-NLP/gte-multilingual-reranker-base` | XLM-RoBERTa cross-encoder | 306M | 8192 | 70+ | Apache 2.0 | `AutoModelForSequenceClassification` |
| 3 | `nvidia/llama-nemotron-rerank-1b-v2` | Llama-3.2 bidirectional cross-encoder | 1B | 8192 | 26 (incl. Indonesian) | NVIDIA Open | `AutoModelForSequenceClassification` |
| 4 | `Qwen/Qwen3-Reranker-0.6B` | Qwen3-0.6B causal LM | 0.6B | 32K | 100+ | Apache 2.0 | `AutoModelForCausalLM` |
| 5 | `jinaai/jina-reranker-v3` | Qwen3-0.6B listwise cross-attention | 0.6B | 131K | 89+ | CC BY-NC 4.0 | `AutoModel` (custom) |
| 6 | **RAG Fusion** | Paraphrase + multi-query + RRF | none | — | — | — | LLM + ChromaDB |

---

## Pipeline Configuration

- **Flow:** Router → Retriever (top-k=20, no threshold) → Reranker (top-n=5) → Generator
- **Reranker gates:** Skipped for `"none"` routes and `"aggregate"` query types
- **GPU:** 1× NVIDIA (8 GB VRAM), CUDA
- **Embedding model:** `BAAI/bge-m3` (1024-dim, loaded separately, not unloaded during reranking)
- **LLM generator:** DeepSeek V4 Pro (temperature 0.3)
- **LLM judge (eval):** DeepSeek V4 Flash (temperature 0, thinking disabled)
- **Test dataset:** 40 questions (mixed EN/ID, 7 lookup + aggregate + none routes)

---

## Aggregate Scores

| Metric | BGE | GTE | Nemotron | Qwen3 | Jina V3 | RAG Fusion |
|---|---|---|---|---|---|---|------|
| **Faithfulness** | 0.91 | **0.95** | 0.84 | 0.87 | 0.96 | 0.93 |
| **Contextual Recall** | 0.82 | 0.75 | **0.85** | 0.70 | 0.77 | **0.88** |
| **Contextual Precision** | 0.53 | 0.53 | **0.56** | 0.36 | 0.57 | 0.56 |
| **G-Eval Correctness** | **0.62** | 0.53 | 0.61 | 0.44 | 0.55 | **0.72** |
| **Router Accuracy** | 75.0% | 77.5% | 80.0% | 77.5% | 72.5% | **82.5%** |

*LLM-as-judge metrics: ±0.05–0.08 run-to-run variance at temperature 0. Within-noise differences should be interpreted cautiously.*

---

## Latency

| Metric | BGE | GTE | Nemotron | Qwen3 | Jina V3 | RAG Fusion |
|---|---|---|---|---|---|---|------|
| Avg reranker per call | 705ms | **503ms** | 7,284ms¹ | 1,342ms | 1,381ms | — (none) |
| w/o 1st call (warmed) | ~500ms | **~370ms** | ~580ms | ~1,200ms | ~900ms | — (none) |
| Avg total per case² | 11,490ms | 13,417ms | 17,306ms | 12,565ms | 13,441ms | **13,067ms** |
| GPU VRAM required³ | ~1.2 GB | ~0.6 GB | ~2 GB | ~1.2 GB | ~1.2 GB | 0 GB |

¹ Nemotron case #0: 165s cold-start model load competing with BGE-M3 on 8 GB GPU. Subsequent calls ~580ms.

² Total includes router (LLM call), embedding, ChromaDB query, reranker, generator (LLM call). DeepEval scoring time excluded.

³ Approximate, in addition to BGE-M3 embedding model (~2 GB).

---

## Per-Model Notes

### 1. BGE V2-M3 (`bge-reranker-v2-m3`)

- **Status:** Baseline reference
- **Strengths:** Same model family as embedder (BGE-M3), well-optimized via FlagEmbedding
- **Weaknesses:** FlagEmbedding dependency (version conflicts with transformers), 512-token truncation
- **CP analysis:** 0.53 — re-ranking within same embedding space means high correlation with cosine order

### 2. GTE Multilingual (`gte-multilingual-reranker-base`)

- **Status:** Recommended for production
- **Strengths:** Fastest (370ms warmed), smallest (306M), 8192-token context, no special dependencies
- **Weaknesses:** Requires `trust_remote_code=True`
- **CP analysis:** 0.53 — identical to BGE despite different architecture and embedding space

### 3. Nemotron 1B (`llama-nemotron-rerank-1b-v2`)

- **Status:** Best CR (0.85), but impractical
- **Strengths:** Bidirectional Llama attention, explicit Indonesian support, commercial license
- **Weaknesses:** 1B params strains 8 GB GPU alongside BGE-M3 (165s cold load), borderline OOM risk
- **CP analysis:** 0.56 — marginally highest CP, but within noise. Not worth the 2× VRAM cost

### 4. Qwen3 (`Qwen3-Reranker-0.6B`)

- **Status:** Failed — not suitable for this dataset
- **Strengths:** 100+ languages, instruction-aware, SOTA on MTEB benchmarks
- **Weaknesses:** Yes/no logit scoring produces coarse rankings for structured indicator data. Default "web search" instruction mismatched for CSV tables and markdown guidelines. CP collapsed to 0.36.
- **Root cause:** Causal LM with binary relevance judgment cannot rank near-equally-relevant chunks within the same domain. Works well for diverse web passage retrieval, fails for homogeneous structured data.

### 5. Jina V3 (`jina-reranker-v3`)

- **Status:** Eliminated — no advantage over BGE
- **Strengths:** Listwise joint attention (64 docs), 131K context, SOTA BEIR
- **Weaknesses:** CC BY-NC 4.0 license (non-commercial), listwise architecture adds latency without CP gain
- **CP analysis:** 0.57 — highest CP, but within noise (LLM judge ±0.05–0.08). License limits deployment.

### 6. RAG Fusion (`paraphrase → multi-query → RRF`)

- **Status:** Replaces the reranker — no external model needed
- **Implementation:** DeepSeek V4 Flash generates 3 paraphrased queries (temp 0.7). Original + 3 variants search ChromaDB (top-k=10 each). Reciprocal Rank Fusion (k=60) merges results. No reranker, no extra GPU memory.
- **Results:** CR 0.88 (+6% over BGE), GE 0.72 (+10%), Router 82.5%. CP 0.56 — within noise of reranker baselines but achieved without one.
- **Latency:** 13,067ms/case — only 14% slower than BGE single-query (11,490ms) despite 4× the embeddings
- **Key insight:** Paraphrasing resolves vocabulary gaps (e.g. "UI GM" → "UI GreenMetric ranking") that no cross-encoder could fix. Case #29 ("When was UI GM created?") went from CR 0.00 / "Sorry, I don't know" to CR 1.00 / correct answer.
- **Remaining CR gaps:** Case #9 and #35 still fail — these are the aggregate-counting query (retriever gives all 118 chunks, no fusion needed) and a misrouted csv-vs-pdf query where the answer isn't in the routed source

---

## Key Findings

### 1. Contextual Precision is bottlenecked by retrieval, not reranking

CP is stuck at ~0.55 across three different cross-encoder architectures (BGE, GTE, Nemotron) **and RAG Fusion** (0.56). The reranker cannot fix what the retriever doesn't find. However, RAG Fusion pushed CR from 0.82 → 0.88, showing that query reformulation does improve recall — just not precision.

### 2. Cross-encoder architecture doesn't matter at this scale

XLMRoberta (BGE), XLM-RoBERTa (GTE), and Llama-3.2 bidirectional (Nemotron) all converge to the same CP. The embedding model (BGE-M3) dominates retrieval quality; the reranker provides marginal polish.

### 3. LLM-based reranking is counterproductive for structured domain data

Qwen3's yes/no token scoring — state-of-the-art on MTEB — performed worst on this dataset (CP 0.36). Structured CSV indicators and markdown guidelines don't benefit from the "web search passage" paradigm that LLM rerankers were trained on.

### 4. RAG Fusion is the best overall approach

Without any reranker, fusion matches CP (0.56) while improving CR (+6%) and GE (+10%). The paraphrasing step resolves vocabulary mismatches that constrained all cross-encoder approaches. At 13s per query with no additional GPU memory, it's practical for production.

### 5. Remaining improvement requires retrieval-level changes

Options to explore:
- Upgrade embedding model from BGE-M3 to a newer multilingual embedder
- Implement hybrid search (dense + sparse/BM25)
- Fine-tune the embedder on domain-specific query-document pairs
- Improve router accuracy for csv-vs-pdf misroutes (cases #10, #35)

---

## Recommendation

**Use RAG Fusion as the default retrieval strategy.** It delivers the best CR (0.88) and GE (0.72) without requiring a reranker, at competitive latency (13s/case). Keep the reranker as an optional opt-in (`RAG_RERANK=1`) for users who want the marginal CP polish from GTE Multilingual.
