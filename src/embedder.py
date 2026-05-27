"""Embedder for the UI GreenMetric RAG system.

Manages the embedding model and provides utilities for encoding text
into vectors for ChromaDB storage and query-time retrieval.
"""

from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = EMBED_MODEL.get_embedding_dimension()

print(f"Embedding model: paraphrase-multilingual-MiniLM-L12-v2")
print(f"Embedding dimension: {EMBED_DIM}")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(texts: list[str], *, show_progress: bool = True) -> list[list[float]]:
    """Encode a batch of text strings into 384-dimensional vectors.

    Uses the loaded ``EMBED_MODEL``.  Input is a list
    of plain strings — no chunk‑wrapping required.  The text should be
    pre‑processed and cleaned as needed before calling this function.

    Parameters:
        texts:          List of chunk ``"content"`` strings to embed.
        show_progress:  Show a tqdm progress bar during encoding
                        (useful for large batches like 300 + chunks).

    Returns:
        list[list[float]]: One embedding vector per input string.  Outer
        list is aligned with *texts* (``len(output) == len(texts)``).
        Each inner list has ``EMBED_DIM`` floats.
    """
    return EMBED_MODEL.encode(texts, show_progress_bar=show_progress).tolist()


# ---------------------------------------------------------------------------
# ChromaDB storage
# ---------------------------------------------------------------------------

def store(source_chunks: dict[str, list[dict]], *, client_path: str = "./chroma_db", collection_name: str = "greenmetric_v05") -> None:
    """Embed every chunk and persist them into a single ChromaDB collection.

    Iterates over every source in *source_chunks*, flattens their
    ``"content"`` strings into one batch, embeds them via :func:`embed`,
    and inserts them together with their ``"metadata"`` dicts and
    sequential per‑source IDs.  The collection uses cosine as its
    distance metric and is created if it does not already exist.

    Parameters:
        source_chunks:  Dict produced by the chunker (or the notebook's
                        ``sources`` variable), mapping source names
                        (``"pdf"``, ``"csv_appendix1"``, ...) to lists of
                        ``{content, metadata}`` dicts.
        client_path:    Filesystem directory for the ChromaDB persistent
                        client (default ``"./chroma_db"``).
        collection_name: Name of the ChromaDB collection to create / reuse
                        (default ``"greenmetric_v05"``).

    Returns:
        None.  Side effect: the collection is populated in ChromaDB at
        *client_path*.  Subsequent calls with the same arguments will
        add chunks with matching IDs.
    """
    client = chromadb.PersistentClient(path=client_path)
    collection = client.get_or_create_collection(
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