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


# ---------------------------------------------------------------------------
# Aggregate stats from metadata (zero LLM)
# ---------------------------------------------------------------------------

def aggregate_stats(
    source: str,
    client_path: str = "./chroma_db",
    collection_name: str = "greenmetric_bgem3",
) -> dict | None:
    """Extract aggregate facts from ChromaDB metadata. No LLM needed.

    Returns a dict of structured stats for the generator, or None
    if the source doesn't support metadata aggregation.
    """
    client = chromadb.PersistentClient(path=client_path)
    collection = client.get_collection(collection_name)
    raw = collection.get(where={"source": source})
    docs = raw.get("documents", []) or []
    metas = raw.get("metadatas", []) or []

    if not docs:
        return None

    if source == "csv_appendix1":
        counts = {}
        max_score = 0
        min_score = float("inf")
        max_options_count = 0
        max_options_meta: dict[str, object] = {}
        evidence_count = 0
        for i, meta in enumerate(metas):
            cat = meta.get("category", "?")
            counts[cat] = counts.get(cat, 0) + 1
            ms = meta.get("max_score", -1)
            if isinstance(ms, (int, float)) and ms > 0:
                max_score = max(max_score, ms)
                min_score = min(min_score, ms)
            if meta.get("evidence_required") == "Yes":
                evidence_count += 1
            doc = docs[i] if i < len(docs) else ""
            opt_count = sum(1 for line in doc.split("\n") if line.strip().startswith("["))
            if opt_count > max_options_count:
                max_options_count = opt_count
                max_options_meta = dict(meta)
        winner_q = max_options_meta.get("question_no", "?") if max_options_meta else "?"
        winner_ev = max_options_meta.get("evidence_required", "?") if max_options_meta else "?"
        winner_cat = max_options_meta.get("category", "?") if max_options_meta else "?"
        stats = (
            f"Aggregate statistics from {sum(counts.values())} UI GreenMetric indicators across 7 categories:\n"
            + "Category counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()) + "\n"
            + f"Maximum single-criterion score: {max_score}\n"
            + f"Minimum single-criterion score: {min_score}\n"
            + f"Most answer options: indicator {winner_q} in {winner_cat} with {max_options_count} options (evidence required: {winner_ev})\n"
            + f"Indicators requiring evidence: {evidence_count} of {sum(counts.values())}"
        )
        return stats

    if source == "csv_table1":
        by_country = {}
        for doc in docs:
            lines = doc.strip().split("\n")
            country = lines[0].replace("Country: ", "").strip() if lines else "?"
            unis = [
                l.strip() for l in lines[2:]
                if l.strip() and not l.startswith("Country:")
            ]
            by_country[country] = by_country.get(country, []) + unis
        stats = (
            f"National coordinators across {len(by_country)} countries:\n"
            + "\n".join(
                f"  {c} ({len(u)}): {', '.join(u)}"
                for c, u in sorted(by_country.items())
            )
        )
        return stats

    if source == "csv_table2":
        weights = []
        for doc in docs:
            if "Category:" in doc and "Weight(%):" in doc:
                cat = doc.split("Category:")[1].split("Weight")[0].strip() if "Category:" in doc else "?"
                wt = doc.split("Weight(%):")[1].strip() if "Weight(%):" in doc else "?"
                try:
                    wt_val = float(wt)
                except ValueError:
                    wt_val = 0
                weights.append((cat, wt_val, wt))
        stats = (
            "Category weight percentages for UI GreenMetric evaluation:\n"
            + "\n".join(f"  {c}: {w}%" for c, _, w in sorted(weights, key=lambda x: -x[1]))
        )
        return stats

    if source == "csv_table4":
        lines = []
        for doc in docs:
            lines.append(doc.strip())
        return "Emission source scopes:\n" + "\n\n".join(lines)

    if source == "csv_appendix2":
        categories = set()
        for doc in docs:
            line = doc.strip().split("\n")[0] if doc else ""
            cat = line.replace("Category: ", "").strip() if "Category:" in line else ""
            if cat:
                categories.add(cat)
        return "Green building element categories: " + ", ".join(sorted(categories))

    if source == "csv_appendix3":
        req_counts = {}
        for doc in docs:
            parts = doc.strip().split("\n") if doc else []
            code = parts[0].replace("Field code: ", "").strip() if parts else "?"
            name = parts[1].replace("Field category: ", "").strip() if len(parts) > 1 else "?"
            reqs = [l.strip() for l in parts[2:] if l.strip() and not l.startswith("Field")]
            req_counts[f"{code} ({name})"] = len(reqs)
        stats = (
            "Smart building requirement counts per field code:\n"
            + "\n".join(f"  {k}: {v} requirements" for k, v in sorted(req_counts.items()))
        )
        return stats

    return None
