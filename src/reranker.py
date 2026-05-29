"""Reranker for the UI GreenMetric RAG system.

Re-scores retrieved chunks with Nemotron-1B — a bidirectional Llama
cross-encoder fine-tuned for multilingual retrieval with explicit
Indonesian support.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model (loaded lazily)
# ---------------------------------------------------------------------------

_model: AutoModelForSequenceClassification | None = None
_tokenizer: AutoTokenizer | None = None
_MODEL_NAME = "nvidia/llama-nemotron-rerank-1b-v2"


def _load() -> None:
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME,
            trust_remote_code=True,
            padding_side="left",
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if _DEVICE == "cuda" else torch.float32,
        )
        _model.to(_DEVICE)
        _model.eval()


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------

_BATCH_SIZE = 4


def rerank(
    query: str,
    chunks: list[dict],
    *,
    top_n: int = 5,
) -> list[dict]:
    if not chunks:
        return []

    _load()

    documents = [chunk["content"] for chunk in chunks]
    all_scores: list[float] = []

    with torch.no_grad():
        for i in range(0, len(documents), _BATCH_SIZE):
            batch_docs = documents[i : i + _BATCH_SIZE]
            texts = [
                f"question:{query} \n \n passage:{doc}"
                for doc in batch_docs
            ]

            inputs = _tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=1024,
            )
            inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

            scores = _model(**inputs).logits.view(-1,).float().cpu().tolist()
            all_scores.extend(scores)

            del inputs
            torch.cuda.empty_cache()

    for chunk, score in zip(chunks, all_scores):
        chunk["rerank_score"] = score

    chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return chunks[:top_n]
