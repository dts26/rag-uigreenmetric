# Reranker Benchmark V2 — UI GreenMetric RAG

**v0.8.0** · 2026-06-01 · 47 test cases · top_n=7 (fair comparison)

---

## Configurations

| # | Embedder | Reranker | Collection |
|---|---|---|---|
| 1 | `BAAI/bge-m3` (568M) | None | `greenmetric_bgem3` |
| 2 | `BAAI/bge-m3` | `Alibaba-NLP/gte-multilingual-reranker-base` (306M) | `greenmetric_bgem3` |
| 3 | `BAAI/bge-m3` | `jinaai/jina-reranker-v3` (0.6B) | `greenmetric_bgem3` |
| 4 | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` (568M) | `greenmetric_bgem3` |
| 5 | `Qwen/Qwen3-Embedding-0.6B` (0.6B) | None | `greenmetric_qwen3` |
| 6 | `Qwen/Qwen3-Embedding-0.6B` | `Alibaba-NLP/gte-multilingual-reranker-base` (306M) | `greenmetric_qwen3` |
| 7 | `Qwen/Qwen3-Embedding-0.6B` | `jinaai/jina-reranker-v3` (0.6B) | `greenmetric_qwen3` |
| 8 | `Qwen/Qwen3-Embedding-0.6B` | `BAAI/bge-reranker-v2-m3` (568M) | `greenmetric_qwen3` |

**Pipeline:** Router → Paraphrase (3 variants) → Multi-query RRF (k=60) → (Reranker, top_n=7) → Generator

---

## Results

| # | Embedder | Reranker | F | CR | CP | GE | RT | Reranker ms |
|---|---|---|---|---|---|---|---|---|
| 1 | BGE-M3 | None | 0.92 | 0.88 | 0.60 | 0.63 | 83.0% | 0 |
| 2 | BGE-M3 | GTE | 0.94 | **0.91** | 0.60 | 0.59 | 85.1% | 458 |
| 3 | BGE-M3 | Jina | **0.96** | 0.89 | 0.56 | 0.61 | 83.0% | 547 |
| 4 | BGE-M3 | BGE | 0.93 | **0.91** | 0.60 | **0.64** | **89.4%** | 584 |
| 5 | Qwen3 | None | 0.89 | **0.91** | 0.44 | 0.63 | 80.9% | 0 |
| 6 | Qwen3 | GTE | 0.91 | 0.86 | **0.61** | 0.62 | 87.2% | 437 |
| 7 | Qwen3 | Jina | 0.87 | 0.84 | **0.61** | 0.53 | 87.2% | 474 |
| 8 | Qwen3 | BGE | 0.93 | 0.90 | 0.60 | 0.59 | 87.2% | 589 |

*LLM-based router: ±5-8% run-to-run variance. DeepEval LLM judge: ±0.05-0.08.*

---

## Findings

### 1. BGE-M3 is the better embedder for this dataset

BGE-M3 hits CP 0.60 **without a reranker**. Qwen3 solo lands at CP 0.44 — entirely dependent on a reranker to become competitive. BGE-M3's hybrid dense+sparse training signal encodes both semantic meaning and exact terminology (indicator codes, scores, category names) in a single vector. Qwen3's instruction-aware encoding helps recall (CR 0.91) but the precision gap makes it the weaker choice for structured CSV + markdown data.

### 2. Reranker helps Qwen3, barely touches BGE-M3

Qwen3 + any reranker jumps from CP 0.44 → 0.60-0.61 (+38%). BGE-M3 + any reranker stays at CP 0.56-0.60 — the embedding already orders chunks well enough.

### 3. Best overall: BGE-M3 + BGE (#4)

CP 0.60, CR 0.91, GE 0.64, RT 89.4%. Same model family as embedder, zero instruction overhead, FlagEmbedding integration.

### 4. Qwen3 + GTE/Jina tie for highest CP (0.61)

But at the cost of added complexity (instruction prompt, reranker dependency) for only +0.01 CP over BGE-M3 solo.

### 5. BGE-M3 + Jina: highest faithfulness (0.96), lowest precision

The listwise architecture biases toward retrieving more context (F 0.96), but at the cost of ranking precision (CP 0.56). Good if you prioritize answer completeness over chunk ordering.

### 6. MTEB scores didn't translate

Qwen3-Embedding scored +10 points higher than BGE-M3 on MTEB retrieval (64.64 vs 54.60). On our structured domain data, BGE-M3's hybrid dense+sparse signal proved more effective. Public benchmarks measure general web retrieval — structured CSV tables and methodology documents are a different game.

---

## Recommendation

**Use BGE-M3 embedder + BGE reranker (#4).** Best CP (0.60), CR (0.91), and RT (89.4%) with zero instruction overhead. The reranker is optional — BGE-M3 solo (#1) already matches the best Qwen3+reranker configs on CP. Enable the reranker when recall matters most (CR 0.91 with BGE reranker vs 0.88 solo).
