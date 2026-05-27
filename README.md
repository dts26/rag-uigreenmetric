# 📚 RAG - Multi Source Information Retrieval
-------------------------------------------------------
##### The README won't be updated until the new version of the system is finished. Thank you for coming here.
-------------------------------------------------------


> A hybrid Retrieval-Augmented Generation (RAG) project created to learn and implement advanced context injection using real LLM APIs (DeepSeek-V4-Flash).

This system intelligently combines unstructured narrative data (PDFs) with highly structured tabular data (CSV) to answer complex queries regarding the UI GreenMetric Guidelines.

## 🛠️ Tech Stack
* **Language:** Python 3.13.13
* **Embedding Model:** `all-MiniLM-L6-v2` (via SentenceTransformers)
* **Vector Database:** ChromaDB
* **Data Processing:** `pandas`, `math`
* **LLM Engine:** OpenAI Client (DeepSeek API)

## ⚙️ Pipeline Architecture

1. **Unstructured Data (PDF):** Extraction → Recursive Text Chunking → Embed → Store in ChromaDB.
2. **Structured Data (CSV):** Indexing → Question-Grouped Chunks (combining criteria, options, and dynamically injected mathematical formulas) → Embed → Store in ChromaDB.
3. **Retrieval & Generation:** User Query → Embed → Cosine Similarity Search → Strict Threshold Filtering (Guardrail) → Context Concatenation → DeepSeek LLM Generation.

## 🧠 Design Decisions & Engineering Choices
* **Why Cosine Similarity?** It perfectly matches the training metric used by the `all-MiniLM-L6-v2` embedding model.
* **Why Question-Grouped CSV Chunks?** Prevents the retriever from fetching partial or orphaned indicators, ensuring the LLM sees the complete context of a scoring criterion.
* **Why Formula Metadata Injection?** Embedding formulas directly into the chunks drastically reduces LLM hallucination when asked calculation-based questions.
* **Why Threshold at 0.7?** Empirically calibrated on this specific dataset to act as a strict guardrail against out-of-context queries.
* **Why Multi-Chunk Context?** Feeding multiple relevant chunks (Top-K) improves the LLM's synthesis capability for questions requiring information from multiple indicators.

## ⚠️ Known Limitations
* **Analytical Queries Fail:** The system cannot reliably answer aggregation questions like *"How many questions are there in total?"*. The RAG architecture only retrieves a subset of the data (Top-K), leading the LLM to provide confidently wrong answers or outright refusal based on incomplete counts.
* **Dataset-Specific Threshold:** The `0.7` distance threshold is tightly coupled to this specific document corpus and embedding model. It requires manual recalibration if applied to different datasets.
* **Language Constraint:** The `all-MiniLM-L6-v2` model is predominantly English-only, making this specific pipeline unsuitable for cross-lingual retrieval without swapping the embedding model.

## 🎓 What I've Learned
* **Fundamentals of RAG:** Gained a deep understanding of the complete RAG workflow, its practical advantages, and its inherent architectural limitations.
* **The Core Tech Stack:** Realized that developing a robust RAG system relies on four inseparable pillars: intelligent chunking algorithms, a specialized embedding model, a vector database, and the LLM generation engine.
* **Garbage In, Garbage Out:** Learned that obsessing over the LLM's performance or parameter size is useless if the quality of the context is poor. Moving forward, my primary focus will always be on optimizing data pipelines (chunking strategies and embedding models), as they are the true heartbeat of any RAG system.