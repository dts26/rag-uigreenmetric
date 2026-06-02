"""Reranker for the UI GreenMetric RAG system.

BGE V2-M3 cross-encoder. Opt-in via RAG_RERANK=1.
"""

from FlagEmbedding import FlagReranker

_reranker: FlagReranker | None = None


def _get_model() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker


def rerank(
    query: str,
    chunks: list[dict],
    *,
    top_n: int = 7,
) -> list[dict]:
    if not chunks:
        return []

    model = _get_model()
    pairs = [[query, chunk["content"]] for chunk in chunks]
    scores = model.compute_score(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return chunks[:top_n]
