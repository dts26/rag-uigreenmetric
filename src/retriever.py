"""Retriever for the UI GreenMetric RAG system.

Queries ChromaDB with source-aware routing driven by router output.
"""

import chromadb
from src.embedder import embed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    route_result: dict,
    *,
    top_k: int = 5,
    threshold: float = 0.7,
    client_path: str = "./chroma_db",
    collection_name: str = "greenmetric_v05",
) -> list[dict]:
    """Retrieve chunks for *query* based on the router's classification.

    Opens a ChromaDB connection, embeds *query*, then dispatches on
    ``route_result["query_type"]``:

    * ``"none"`` — returns an empty list immediately (no retrieval).
    * ``"lookup"`` — semantic search filtered to the source(s) specified
      by *route_result*, with top‑k results post‑filtered by *threshold*.
    * ``"both"`` — two parallel semantic searches (pdf + csv_source),
      concatenated and post‑filtered by *threshold*.
    * ``"aggregate"`` — fetches **all** chunks for the relevant source
      via an exact metadata filter.  No cosine-distance threshold is
      applied because the retrieval is deterministic, not
      similarity‑based.

    Parameters:
        query:            The user's question (already classified by the
                          router).
        route_result:     Dict from :func:`router.route` with keys
                          ``"source"``, ``"csv_source"``, and
                          ``"query_type"``.
        top_k:            Maximum results returned by each semantic‑search
                          call (``"lookup"`` and ``"both"`` paths only).
        threshold:        Cosine-distance cut‑off, inclusive.  Only used
                          for ``"lookup"`` and ``"both"``.
        client_path:      Filesystem directory for the ChromaDB persistent
                          client (default ``"./chroma_db"``).
        collection_name:  Name of the ChromaDB collection to query
                          (default ``"greenmetric_v05"``).

    Returns:
        list[dict]: Each dict has keys ``"content"`` (str),
        ``"metadata"`` (dict), and ``"distance"`` (float).  Sorted
        ascending by distance.  ``"aggregate"`` results have
        ``"distance": 0.0``.  An empty list is returned when the router
        says ``"none"`` or when no results pass the threshold.
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
            query, {"source": "pdf"}, top_k, threshold, collection
        )
        csv_results = _semantic_search(
            query, {"source": csv_source}, top_k, threshold, collection
        )
        return _sort_by_distance(pdf_results + csv_results)

    if query_type == "aggregate":
        agg_source = csv_source if csv_source else source
        return _fetch_all({"source": agg_source}, collection)

    lookup_source = csv_source if csv_source else source

    return _semantic_search(
        query, {"source": lookup_source}, top_k, threshold, collection
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _semantic_search(
    query: str,
    where: dict,
    top_k: int,
    threshold: float,
    collection,
) -> list[dict]:
    """Embed *query*, run ChromaDB semantic search, post‑filter by threshold."""
    query_vector = embed([query])
    raw = collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
        where=where,
    )
    results = []
    for i in range(len(raw["documents"][0])):
        distance = raw["distances"][0][i]
        if distance <= threshold:
            results.append({
                "content": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": distance,
            })
    return results


def _fetch_all(where: dict, collection) -> list[dict]:
    """Fetch every chunk matching *where* via exact metadata lookup.

    No embedding, no semantic search, no threshold — deterministic
    retrieval.  Used for aggregate queries that need the full dataset.
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
