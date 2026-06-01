"""Reranker for the UI GreenMetric RAG system.

Re-scores retrieved chunks with Jina V3 — a listwise cross-encoder
that processes all documents jointly with causal self-attention.
"""

import torch
from transformers import AutoModel

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_reranker: AutoModel | None = None


def _get_model() -> AutoModel:
    global _reranker
    if _reranker is None:
        _reranker = AutoModel.from_pretrained(
            "jinaai/jina-reranker-v3",
            torch_dtype=torch.bfloat16 if _DEVICE == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        _reranker.to(_DEVICE)
        _reranker.eval()
    return _reranker


def rerank(query: str, chunks: list[dict], *, top_n: int = 5) -> list[dict]:
    if not chunks:
        return []
    model = _get_model()
    documents = [chunk["content"] for chunk in chunks]
    results = model.rerank(query, documents, top_n=min(top_n, len(documents)))
    reranked = []
    for result in results:
        chunk = chunks[result["index"]].copy()
        chunk["rerank_score"] = float(result["relevance_score"])
        reranked.append(chunk)
    return reranked
