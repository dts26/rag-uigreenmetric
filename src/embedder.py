"""Embedder for the UI GreenMetric RAG system.

Manages the BGE-M3 embedding model and provides utilities for encoding
text into vectors for ChromaDB storage and query-time retrieval.
"""

from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

EMBED_MODEL = SentenceTransformer("BAAI/bge-m3")
EMBED_DIM = EMBED_MODEL.get_embedding_dimension()

print(f"Embedding model: BGE-M3")
print(f"Embedding dimension: {EMBED_DIM}")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(texts: list[str], *, show_progress: bool = True) -> list[list[float]]:
    """Encode document/chunk text."""
    return EMBED_MODEL.encode(
        texts, show_progress_bar=show_progress, batch_size=8
    ).tolist()


def embed_query(texts: list[str], *, show_progress: bool = True) -> list[list[float]]:
    """Encode search queries. BGE-M3 doesn't need instruction prefix."""
    return EMBED_MODEL.encode(
        texts, show_progress_bar=show_progress, batch_size=8
    ).tolist()


# ---------------------------------------------------------------------------
# ChromaDB storage
# ---------------------------------------------------------------------------

def store(
    source_chunks: dict[str, list[dict]],
    *,
    client_path: str = "./chroma_db",
    collection_name: str = "greenmetric_bgem3",
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
