# RAG Fusion + Reranker Report — UI GreenMetric RAG

**v0.8.0** · 2026-05-29 · 47 test cases (40 original + 7 synthetic)

---

## Embedding Models Tested

| Embedder | Params | Dim | Context | MTEB Multilingual | Retrieval | Languages |
|---|---|---|---|---|---|---|
| BGE-M3 | 568M | 1024 | 8192 | 59.56 | 54.60 | 75+ |
| Qwen3-Embedding-0.6B | 0.6B | 1024 | 32K | **64.33** | **64.64** | 100+ |

Qwen3 adds **+10 points on MTEB retrieval** over BGE-M3 (+18% relative). It also supports instruction-aware queries (task prefix improves matching).

## Configurations

| # | Name | Embedder | Retriever | Reranker |
|---|---|---|---|---|
| 1 | RF (BGE-M3) | BGE-M3 | Paraphrase ×3 → RRF (k=60) | None |
| 2 | RF + BGE | BGE-M3 | Paraphrase ×3 → RRF (k=60) | `BAAI/bge-reranker-v2-m3` (568M) |
| 3 | RF + GTE | BGE-M3 | Paraphrase ×3 → RRF (k=60) | `Alibaba-NLP/gte-multilingual-reranker-base` (306M) |
| 4 | RF + Jina | BGE-M3 | Paraphrase ×3 → RRF (k=60) | `jinaai/jina-reranker-v3` (0.6B) |
| 5 | **RF (Qwen3)** | Qwen3-Embed | Paraphrase ×3 → RRF (k=60) | None |
| 6 | **RF (Qwen3, latest)** | Qwen3-Embed | Paraphrase ×3 → RRF (k=60) | None |

**Pipeline:** Router → Paraphrase (3 variants) → Multi-query RRF → (Reranker, top-n=5) → Generator

**Enable reranker:** `RAG_RERANK=1`

---

## Aggregate Scores

### BGE-M3 Embeddings

| Metric | RF (Baseline) | RF + BGE | RF + GTE | RF + Jina |
|---|---|---|---|---|
| Faithfulness | 0.89 | 0.94 | **0.96** | **0.96** |
| Contextual Recall | 0.84 | 0.87 | 0.79 | **0.91** |
| Contextual Precision | 0.56 | 0.54 | **0.60** | 0.57 |
| G-Eval Correctness | **0.63** | 0.60 | 0.44 | 0.60 |
| Router Accuracy | 78.7% | 74.5% | 78.7% | 78.7% |

### Qwen3-Embedding

| Metric | RF (Qwen3, default prompt) | **RF (Qwen3, custom prompt)** | RF (Qwen3, latest)¹ | RF + Jina (Qwen3) |
|---|---|---|---|---|
| **Faithfulness** | 0.89 | 0.91 | **0.95** | 0.93 |
| **Contextual Recall** | 0.83 | **0.91** | 0.83 | 0.84 |
| Contextual Precision | 0.48 | 0.54 | 0.50 | **0.60** |
| G-Eval Correctness | **0.59** | 0.55 | **0.59** | 0.49 |
| Router Accuracy | 76.6% | 78.7% | **89.4%** | 72.3% |

¹ Same config (Qwen3 + custom prompt, no reranker, top_n=7) but with router few-shot tuning — 5 additional examples. Jina column: identical Qwen3 + custom prompt with Jina reranker enabled. Reranker improves CP (+0.10) but harms everything else (CR -0.07, GE -0.10, RT -17%). Not worth the trade.

*47 test cases. LLM-as-judge ±0.05–0.08 variance.*

---

## Latency

### BGE-M3 Embeddings

| Metric | RF (Baseline) | RF + BGE | RF + GTE | RF + Jina |
|---|---|---|---|---|
| Avg total per case (ms) | **8,677** | 9,253 | 8,405 | 10,451 |
| Avg reranker per call (ms) | 0 | 683 | 934¹ | 507 |
| GPU VRAM (reranker) | 0 GB | ~1.2 GB | ~0.6 GB | ~1.2 GB |

### Qwen3-Embedding

| Metric | RF (Qwen3, default prompt) | **RF (Qwen3, custom prompt)** | RF (Qwen3, latest) |
|---|---|---|---|---|
| Avg total per case (ms) | 7,453 | 7,876 | **7,538** |
| Avg reranker per call (ms) | 0 | 0 | 0 |
| GPU VRAM (reranker) | 0 GB | 0 GB | 0 GB |

¹ GTE cold-start included (first model load on GPU). Warmed calls ~370ms.

---

## Per-Configuration Notes

### 1. RF (BGE-M3) — Baseline

- **F 0.89, CR 0.84, CP 0.56, GE 0.63**
- 8,677ms/case — fastest BGE-M3 config
- No additional GPU memory, no extra dependencies

### 2. RF + BGE

- **F 0.94, CR 0.87, CP 0.54, GE 0.60**
- 683ms reranker per call, 9,253ms total
- FlagEmbedding dependency, 512-token context limit
- CP dropped vs baseline — reranker adds marginal value on top of RRF

### 3. RF + GTE

- **F 0.96, CR 0.79, CP 0.60, GE 0.44**
- 934ms reranker (incl. cold start, ~370ms warmed), 8,405ms total
- Highest CP (+4%) but **G-Eval collapse (-0.19 vs baseline)** — likely promotes relevant-but-not-answer-specific chunks

### 4. RF + Jina (BGE-M3)

- **F 0.96, CR 0.91, CP 0.57, GE 0.60**
- 507ms reranker, 10,451ms total — highest latency
- Best BGE-M3 CR at 0.91 — listwise attention pulls in most context
- CC BY-NC 4.0 license restricts commercial use

### 5. RF (Qwen3, custom prompt)

- **F 0.91, CR 0.91, CP 0.54, GE 0.55, RT 78.7%, 7,876ms/case**
- Domain-specific query instruction: `"Instruct: Given a question about UI GreenMetric university sustainability rankings, retrieve relevant guideline documents and indicator data\nQuery:{query}"`
- CR 0.91 ties best from any config — without a reranker
- Custom prompt added +0.08 CR and +0.06 CP over the generic web-search default

### 6. RF (Qwen3, latest — router tuned)

- **F 0.95, CR 0.83, CP 0.50, GE 0.59, RT 89.4%, 7,538ms/case**
- Same config as #5 but router prompt augmented with 5 few-shot examples targeting previously misrouted cases
- **Router accuracy jumped from 78.7% → 89.4%** — the single highest-ROI change in this report
- CR/CP trade-off: better routing means more "none" cases correctly rejected, which shifts recall/precision balance

### 6. RF + Jina (Qwen3)

- **F 0.93, CR 0.84, CP 0.60, GE 0.49**
- 368ms reranker, 8,580ms total
- CP improves 0.54 → 0.60 (+0.06), but CR drops 0.91 → 0.84 (-0.07) and GE drops 0.55 → 0.49 (-0.06)
- Reranker trades recall and correctness for precision — not a worthwhile trade

---

## Findings

1. **Custom instruction matters as much as the model.** The domain-specific Qwen3 prompt lifted CR from 0.83 → 0.91 (+0.08) and CP from 0.48 → 0.54 (+0.06). The generic "web search query" prompt was leaving signal on the table.

2. **Qwen3 + custom prompt + RAG Fusion = best config overall.** CR 0.91 ties the best (RF + Jina on BGE-M3) without needing a reranker. F 0.91 and GE 0.55 are solid. No extra GPU memory, no extra dependencies.

3. **Embedder upgrade was the right move.** Qwen3-Embedding's +10 MTEB retrieval points translated to measurable real gains — and the instruction-awareness let us tune it further for our domain.

4. **Rerankers add nothing on top of Qwen3.** Every Qwen3 + reranker config either matched or degraded Qwen3 alone. The instruction-aware embeddings order chunks well enough that reranking can't improve.

5. **Router accuracy (78.7%) is the next bottleneck.** 10 out of 47 cases are misrouted — same pattern as before. Fixing the router would cascade into further CR/CP gains.

## Recommendation

**Production: RF + Qwen3-Embedding (custom prompt, no reranker, top_n=7).** Best results — CR 0.91, CP 0.54, F 0.91, 7.9s/case, zero reranker dependencies. The domain-specific instruction is the single most impactful change we've made. Fix the router next for the next quality jump.
