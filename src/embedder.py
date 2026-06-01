"""Embedder for the UI GreenMetric RAG system.

Manages the embedding model and provides utilities for encoding text
into vectors for ChromaDB storage and query-time retrieval.

Auto-detects environment:
    Local/GitHub  → local Qwen3-Embedding via SentenceTransformers
    HF Spaces     → HF Inference API (GPU) when EMBED_BACKEND=hf_api
"""

import os
import numpy as np
import chromadb

# ---------------------------------------------------------------------------
# Query instruction
# ---------------------------------------------------------------------------

_QUERY_INSTRUCTION = (
    "Instruct: Given a question about UI GreenMetric university sustainability "
    "rankings, retrieve relevant guideline documents and indicator data\nQuery:"
)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_BACKEND = os.getenv("EMBED_BACKEND", "local")
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print(f"Embedding model: Qwen3-Embedding-0.6B (local)")
        print(f"Embedding dimension: {_local_model.get_embedding_dimension()}")
    return _local_model


def _embed_local(texts: list[str], instruct: bool = False) -> list[list[float]]:
    """Embed via local Qwen3 SentenceTransformer."""
    model = _get_local_model()
    if instruct:
        return model.encode(
            texts, prompt=_QUERY_INSTRUCTION, show_progress_bar=False, batch_size=4
        ).tolist()
    return model.encode(
        texts, show_progress_bar=False, batch_size=4
    ).tolist()


def _embed_hf_api(texts: list[str], instruct: bool = False) -> list[list[float]]:
    """Embed via HF Inference API (GPU-backed, serverless)."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(model="Qwen/Qwen3-Embedding-0.6B")

    if instruct:
        texts = [f"{_QUERY_INSTRUCTION} {t}" for t in texts]

    result = client.feature_extraction(texts)
    # feature_extraction returns a list of vectors (numpy arrays or lists)
    if hasattr(result[0], "tolist"):
        return [r.tolist() for r in result]
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(texts: list[str], *, show_progress: bool = True) -> list[list[float]]:
    """Encode document/chunk text — no instruction prefix needed."""
    if _BACKEND == "hf_api":
        return _embed_hf_api(texts, instruct=False)
    return _embed_local(texts, instruct=False)


def embed_query(texts: list[str], *, show_progress: bool = True) -> list[list[float]]:
    """Encode search queries with task instruction for better retrieval."""
    if _BACKEND == "hf_api":
        return _embed_hf_api(texts, instruct=True)
    return _embed_local(texts, instruct=True)


# ---------------------------------------------------------------------------
# ChromaDB storage
# ---------------------------------------------------------------------------

def store(
    source_chunks: dict[str, list[dict]],
    *,
    client_path: str = "./chroma_db",
    collection_name: str = "greenmetric_qwen3",
) -> None:
    """Embed every chunk and persist them into a single ChromaDB collection."""
    client = chromadb.PersistentClient(path=client_path)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    all_texts = []
    all_metadatas = []
    all_ids = []

    for source_name, chunk_list in source_chunks.items():
        for i, chunk in enumerate(chunk_list):
            all_texts.append(chunk["content"])
            all_metadatas.append({
                k: ("" if v is None else v)
                for k, v in chunk["metadata"].items()
            })
            all_ids.append(f"{source_name}_chunk_{i}")

    embeddings = embed(all_texts)

    collection.add(
        documents=all_texts,
        metadatas=all_metadatas,
        embeddings=embeddings,
        ids=all_ids,
    )
