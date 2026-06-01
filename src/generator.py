"""Generator for the UI GreenMetric RAG system.

Formats retrieved context, injects conversation history, and calls
DeepSeek V4 Pro to produce the final answer.  Low-confidence detection
appends a warning footer for answers near the relevance threshold.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GENERATOR_CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

GENERATOR_SYSTEM_PROMPT = """You are a UI GreenMetric AI assistant.
You answer questions about the UI GreenMetric Sustainable University
Rankings based ONLY on the provided context.  Do not use any external
information.

If the answer is not found in the context, respond with "Sorry, I don't
know." and do not provide additional information.

Always provide a concise and accurate answer based on the context.
Do not include information that is not explicitly stated in the context."""

LOW_CONFIDENCE_PROMPT = """NOTE: The retrieved context scored close to the
relevance threshold.  Be cautious and qualify your answer where
appropriate."""

LOW_CONFIDENCE_FOOTER = """---
Note: I have low confidence in this answer. The retrieved information was
close to the cosine distance threshold (0.6), so the answer may not be
fully accurate."""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(
    query: str,
    context: list[dict],
    *,
    conversation_history: list[dict] | None = None,
    query_type: str = "lookup",
) -> str:
    """Generate an answer from retrieved context chunks.

    Formats each chunk with a ``=== CONTEXT (source, chunk_type) ===``
    header, injects prior conversation history (up to 7 turns, already
    truncated by the caller), calls DeepSeek-V4-Pro at temperature 0.3,
    and returns the answer string.

    **Low-confidence detection** — if the **top** chunk (lowest distance,
    therefore the best match) has a cosine distance greater than 0.6 AND
    *query_type* is not ``"aggregate"``, the system prompt is hardened
    with :data:`LOW_CONFIDENCE_PROMPT` to make the model more cautious.
    Aggregate queries always have ``distance == 0.0`` (retrieved via exact
    metadata match in ``_fetch_all``) so the check is skipped.
    The pipeline's UI layer is responsible for displaying a warning to
    the user — :func:`generate` does not append any footer text.

    ``"none"`` route queries are caught by the pipeline before this
    function is called — **generate** should never receive them.

    Parameters:
        query:                The user's question.
        context:              List of chunk dicts from
                              :func:`retriever.retrieve`.  Each dict has
                              ``"content"`` (str), ``"metadata"`` (dict
                              with ``"source"`` and ``"chunk_type"``),
                              and ``"distance"`` (float).
        conversation_history: Prior user/assistant message pairs.  Each
                              dict has ``"role"`` and ``"content"``.
                              Capped at 7 messages by the caller.
        query_type:           ``"lookup"`` (default) or ``"aggregate"``.
                              Controls whether low-confidence detection
                              is active (skipped for aggregate).

    Returns:
        tuple[str, int]: The generated answer and the token count from the API.
    """
    low_confidence = _is_low_confidence(context, query_type)

    system_content = GENERATOR_SYSTEM_PROMPT
    if low_confidence:
        system_content += "\n\n" + LOW_CONFIDENCE_PROMPT

    messages = [{"role": "system", "content": system_content}]

    if conversation_history:
        messages.extend(conversation_history)

    context_block = _format_context(context)
    user_content = f"{context_block}\n\n=== QUESTION ===\n{query}" if context_block else query
    messages.append({"role": "user", "content": user_content})

    response = GENERATOR_CLIENT.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        temperature=0.3,
    )

    tokens = getattr(response.usage, "total_tokens", 0)
    answer = response.choices[0].message.content.strip()

    return answer, tokens


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_low_confidence(context: list[dict], query_type: str) -> bool:
    """Return True if the top chunk's distance exceeds the 0.6 warning threshold.

    Aggregate queries are excluded — their chunks are fetched via exact
    metadata match and always have ``distance == 0.0``.
    """
    if not context or query_type == "aggregate":
        return False
    return context[0]["distance"] > 0.6


def _format_context(context: list[dict]) -> str:
    """Format retrieved chunks into labelled context blocks."""
    blocks = []
    for chunk in context:
        src = chunk["metadata"]["source"]
        ctype = chunk["metadata"]["chunk_type"]
        blocks.append(
            f"=== CONTEXT (source: {src}, chunk_type: {ctype}) ===\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(blocks)
