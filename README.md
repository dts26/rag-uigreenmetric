# 📚 RAG — Multi Source Information Retrieval

> A hybrid Retrieval-Augmented Generation (RAG) project created to learn and implement advanced context injection using real LLM APIs (DeepSeek).

This system intelligently combines unstructured narrative data with highly structured tabular data (CSV) to answer complex queries regarding the UI GreenMetric Guidelines.

**Current version: v0.5.1**

## 🛠️ Tech Stack
* **Language:** Python 3.12.13
* **Embedding Model:** `paraphrase-multilingual-MiniLM-L12-v2` (via SentenceTransformers)
* **Vector Database:** ChromaDB
* **Data Processing:** `pandas`, `openpyxl`
* **LLM Engine:** OpenAI Client (DeepSeek API)
* **Evaluation Framework:** DeepEval

## ⚙️ Pipeline Architecture

1. **Unstructured Data:** Markdown → Heading-Level Chunking → Embed → Store in ChromaDB.
2. **Structured Data (CSV):** Question-Grouped Chunks (combining criteria, options, and dynamically injected mathematical formulas) → Embed → Store in ChromaDB.
3. **Retrieval & Generation:** User Query → Router (LLM) → Cosine Similarity Search → Strict Threshold Filtering (Guardrail) → Context Concatenation → DeepSeek LLM Generation.
4. **Router Agent:** Classifies queries by source (PDF, CSV, Both, None) and query type (lookup, aggregate) before retrieval.

## 📊 Evaluation Metrics (DeepEval)

| Metric | What it measures |
|---|---|
| **Faithfulness** | Whether the generated answer is factually supported by the retrieved context. |
| **Contextual Recall** | Whether the retrieved context contains all the information needed to answer the question. |
| **Contextual Precision (NDCG@K)** | Whether relevant chunks are ranked higher in the retrieved results. |
| **G-Eval (Answer Correctness)** | How well the generated answer matches the ground truth (LLM-as-judge, 0–1). |
| **Router Accuracy** | Whether the pipeline router correctly identifies the expected data source. |

## 🧠 Design Decisions & Engineering Choices
* **Why Cosine Similarity?** It perfectly matches the training metric used by the embedding model.
* **Why Question-Grouped CSV Chunks?** Prevents the retriever from fetching partial or orphaned indicators, ensuring the LLM sees the complete context of a scoring criterion.
* **Why Formula Metadata Injection?** Embedding formulas directly into the chunks drastically reduces LLM hallucination when asked calculation-based questions.
* **Why Threshold at 0.7?** Empirically calibrated on this specific dataset to act as a strict guardrail against out-of-context queries (chunks above 0.7 cosine distance are invisible to the LLM).
* **Why Multi-Chunk Context?** Feeding multiple relevant chunks (Top-K=5) improves the LLM's synthesis capability for questions requiring information from multiple indicators.

## ⚠️ Known Limitations
* **Dataset-Specific Threshold:** The `0.7` distance threshold is tightly coupled to this specific document corpus and embedding model. It requires manual recalibration if applied to different datasets.
* **Router Accuracy ~80%:** The router occasionally misclassifies `both`-source queries as single-source, or `pdf` queries as `csv`, causing retrieval failures on edge cases.
* **Aggregate Queries Use Brute-Force:** The retriever fetches all chunks for the source via exact metadata match rather than using a parent-child chunk hierarchy.

## 🎓 What I've Learned
* **Fundamentals of RAG:** Gained a deep understanding of the complete RAG workflow, its practical advantages, and its inherent architectural limitations.
* **The Core Tech Stack:** Realized that developing a robust RAG system relies on four inseparable pillars: intelligent chunking algorithms, a specialized embedding model, a vector database, and the LLM generation engine.
* **Garbage In, Garbage Out:** Learned that obsessing over the LLM's performance or parameter size is useless if the quality of the context is poor. Moving forward, my primary focus will always be on optimizing data pipelines (chunking strategies and embedding models), as they are the true heartbeat of any RAG system.
* **Evaluation is Critical:** Switched from RAGAS to DeepEval after hitting async embedding and instructor JSON truncation issues with the custom LLM judge setup. DeepEval's cleaner API eliminated the need for embedding wrappers and deprecated import patches.
