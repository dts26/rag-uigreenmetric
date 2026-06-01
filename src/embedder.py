"""Embedder for the UI GreenMetric RAG system.

Manages the embedding model and provides utilities for encoding text
into vectors for ChromaDB storage and query-time retrieval.
"""

from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

EMBED_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
EMBED_DIM = EMBED_MODEL.get_embedding_dimension()

print(f"Embedding model: Qwen3-Embedding-0.6B")
print(f"Embedding dimension: {EMBED_DIM}")

# ---------------------------------------------------------------------------
# Query instruction (customized for our domain)
# ---------------------------------------------------------------------------

_QUERY_INSTRUCTION = (
    "Instruct: Given a question about UI GreenMetric university sustainability "
    "rankings, retrieve relevant guideline documents and indicator data\nQuery:"
)


# ---------------------------------------------------------------------------
# Embedding — documents (no instruction prefix)
# ---------------------------------------------------------------------------

def embed(texts: list[str], *, show_progress: bool = True) -> list[list[float]]:
    """Encode document/chunk text — no instruction prompt needed."""
    return EMBED_MODEL.encode(
        texts, show_progress_bar=show_progress, batch_size=4
    ).tolist()


# ---------------------------------------------------------------------------
# Embedding — queries (with instruction prompt)
# ---------------------------------------------------------------------------

def embed_query(texts: list[str], *, show_progress: bool = True) -> list[list[float]]:
    """Encode search queries with task instruction for better retrieval."""
    return EMBED_MODEL.encode(
        texts, prompt=_QUERY_INSTRUCTION, show_progress_bar=show_progress, batch_size=4
    ).tolist()


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
    # Drop old collection if it exists to rebuild cleanly
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
