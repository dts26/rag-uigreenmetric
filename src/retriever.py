"""Retriever for the UI GreenMetric RAG system.

Queries ChromaDB with source-aware routing driven by router output.
"""

import os
import chromadb
from src.embedder import embed_query


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    route_result: dict,
    *,
    top_k: int = 20,
    client_path: str = "./chroma_db",
    collection_name: str = "greenmetric_bgem3",
) -> list[dict]:
    collection_name = os.getenv("RAG_COLLECTION", collection_name)
    """Retrieve chunks for *query* based on the router's classification.

    Opens a ChromaDB connection, embeds *query*, then dispatches on
    ``route_result["query_type"]``:

    * ``"none"`` — returns an empty list immediately (no retrieval).
    * ``"lookup"`` — semantic search filtered by metadata source.
    * ``"both"`` — two parallel semantic searches (pdf + csv_source),
      concatenated and sorted by distance.
    * ``"aggregate"`` — fetches **all** chunks for the relevant source
      via an exact metadata filter (deterministic, no similarity check).

    Parameters:
        query:            The user's question.
        route_result:     Dict from :func:`router.route` with keys
                          ``"source"``, ``"csv_source"``, and
                          ``"query_type"``.
        top_k:            Maximum results returned by each semantic‑search
                          call (``"lookup"`` and ``"both"`` paths only).
        client_path:      ChromaDB persistent client directory.
        collection_name:  ChromaDB collection name.

    Returns:
        list[dict]: Each dict has keys ``"content"`` (str),
        ``"metadata"`` (dict), and ``"distance"`` (float).  Sorted
        ascending by distance.
    """
    source = route_result["source"]
    csv_source = route_result.get("csv_source")
    query_type = route_result.get("query_type", "lookup")

    client = chromadb.PersistentClient(path=client_path)
    collection = client.get_collection(collection_name)

    if source == "none":
        return []

    if source == "both":
        pdf_results = _semantic_search(
            query, {"source": "pdf"}, top_k, collection
        )
        csv_results = _semantic_search(
            query, {"source": csv_source}, top_k, collection
        )
        return _sort_by_distance(pdf_results + csv_results)

    if query_type == "aggregate":
        agg_source = csv_source if csv_source else source
        return _fetch_all({"source": agg_source}, collection)

    lookup_source = csv_source if csv_source else source

    return _semantic_search(
        query, {"source": lookup_source}, top_k, collection
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _semantic_search(
    query: str,
    where: dict,
    top_k: int,
    collection,
) -> list[dict]:
    """Embed *query*, run ChromaDB semantic search, return all top‑k results."""

    query_vector = embed_query([query])
    raw = collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
        where=where,
    )
    results = []
    for i in range(len(raw["documents"][0])):
        distance = raw["distances"][0][i]
        results.append({
            "content": raw["documents"][0][i],
            "metadata": raw["metadatas"][0][i],
            "distance": distance,
        })
    return results


def _fetch_all(where: dict, collection) -> list[dict]:
    """Fetch every chunk matching *where* via exact metadata lookup.

    Deterministic retrieval, Used for aggregate queries that need 
    the full dataset.
    """
    raw = collection.get(where=where)
    results = []
    for i in range(len(raw["documents"])):
        results.append({
            "content": raw["documents"][i],
            "metadata": raw["metadatas"][i],
            "distance": 0.0,
        })
    return results


def _sort_by_distance(results: list[dict]) -> list[dict]:
    """Sort *results* in-place by ascending ``"distance"``."""
    results.sort(key=lambda r: r["distance"])
    return results


# ---------------------------------------------------------------------------
# Multi-query retrieval + Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

_RRF_K = 60


def retrieve_multi(
    queries: list[str],
    route_result: dict,
    *,
    top_k: int = 10,
    client_path: str = "./chroma_db",
    collection_name: str = "greenmetric_bgem3",
) -> list[dict]:
    collection_name = os.getenv("RAG_COLLECTION", collection_name)
    """Multi-query retrieval with Reciprocal Rank Fusion.

    Runs semantic search for each query variant (original + paraphrases),
    then merges results via RRF to produce a unified ranked list.

    Parameters:
        queries:          List of query strings (original + paraphrased).
        route_result:     Dict from :func:`router.route`.
        top_k:            Max results per query variant.
        client_path:      ChromaDB persistent client directory.
        collection_name:  ChromaDB collection name.

    Returns:
        list[dict]: Merged chunks sorted by RRF score descending.
    """
    source = route_result["source"]
    csv_source = route_result.get("csv_source")

    client = chromadb.PersistentClient(path=client_path)
    collection = client.get_collection(collection_name)

    # Build list of (metadata_filter) per search
    if source == "both":
        filters = [{"source": "pdf"}, {"source": csv_source}]
    else:
        lookup = csv_source if csv_source else source
        filters = [{"source": lookup}]

    # Run all searches: queries × filters
    from collections import defaultdict
    chunk_scores: dict[str, float] = defaultdict(float)
    chunk_data: dict[str, dict] = {}

    for q in queries:
        for f in filters:
            results = _semantic_search(q, f, top_k, collection)
            for rank, r in enumerate(results):
                cid = r["metadata"].get("chunk_id", r["content"][:80])
                chunk_scores[cid] += 1.0 / (_RRF_K + rank + 1)
                chunk_data[cid] = r

    merged = []
    for cid, score in chunk_scores.items():
        data = chunk_data[cid].copy()
        data["rrf_score"] = score
        merged.append(data)

    merged.sort(key=lambda r: r["rrf_score"], reverse=True)
    return merged
