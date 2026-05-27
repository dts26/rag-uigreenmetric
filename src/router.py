"""Router for the UI GreenMetric RAG system.

Classifies user queries into data-source categories to guide retrieval
in ChromaDB.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ROUTER_CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

ROUTER_SYSTEM_PROMPT = """You are a query router for the UI GreenMetric RAG system. Your job is to classify a user's question into the correct data source(s) needed to answer it.

=== SOURCE REGISTRY ===

pdf — The UI GreenMetric guidelines narrative document. Contains: what the ranking is, objectives, how to participate, benefits, methodology history, scoring logic, category descriptions, tie-breaking rules, data submission instructions, evidence requirements, network and coordinator information, and detailed definitions for each questionnaire indicator.

csv_appendix1 — Questionnaire master table. 118 scored indicators across 7 categories (SI, EC, WS, WR, TR, ED, GD). Each row has: criteria text, answer options with calculated scores, max_score, indicator code, evidence_required flag, and colored marker.

csv_appendix2 — Green building elements (GBI). 6 element categories each listing sub-categories and specific elements/requirements for both existing non-residential buildings and new construction (NRNC).

csv_appendix3 — Smart building requirements. 6 field codes (B=Automation, S=Safety, E=Energy, A=Water, I=Indoor environment, L=Lighting). Each field lists requirement codes with names and descriptions.

csv_table1 — National coordinators. 35 universities across 30 countries.

csv_table2 — Category weighting. 7 categories with their percentage of total points.

csv_table4 — Greenhouse gas emission sources. Categorized by Scope 1 (Direct), Scope 2 (Indirect), and Scope 3 (Other indirect) with emission source names and descriptions.

=== FEW-SHOT EXAMPLES ===

Q: "What kind of evidence does UI GreenMetric accept for data submission?"
A: {"source": "pdf", "csv_source": null, "query_type": "lookup"}

Q: "Berapa universitas yang mendaftar pada edisi pertama UI GreenMetric?"
A: {"source": "pdf", "csv_source": null, "query_type": "lookup"}

Q: "Pertanyaan 2.1. ada nilainya gak?"
A: {"source": "csv", "csv_source": "csv_appendix1", "query_type": "lookup"}

Q: "Ada berapa total indikator yang dimiliki kategori SI dan berapa yang butuh evidence?"
A: {"source": "csv", "csv_source": "csv_appendix1", "query_type": "aggregate"}

Q: "How many answer options does the indicator 1.2 provide?"
A: {"source": "csv", "csv_source": "csv_appendix1", "query_type": "lookup"}

Q: "Refrigerant leaks from AC equipment fall under which emission scope?"
A: {"source": "csv", "csv_source": "csv_table4", "query_type": "lookup"}

Q: "The guidelines mention that open space ratio affects scoring, what is the actual indicator code for that and what ratio gives the highest score?"
A: {"source": "both", "csv_source": "csv_appendix1", "query_type": "lookup"}

Q: "Pedoman menyebutkan gedung hijau sebagai salah satu elemen penilaian. Lalu di lampiran 2, elemen apa saja yang termasuk untuk bangunan baru di bawah Energy Efficiency?"
A: {"source": "both", "csv_source": "csv_appendix2", "query_type": "lookup"}

Q: "The text describes smart building indicators under Energy and Climate Change. According to the requirements appendix, what systems count as smart building automation?"
A: {"source": "both", "csv_source": "csv_appendix3", "query_type": "lookup"}

Q: "How many countries are represented and which one has the most coordinators?"
A: {"source": "both", "csv_source": "csv_table1", "query_type": "aggregate"}

Q: "According to the methodology section, universities must calculate their carbon footprint. Based on the emission sources table, which Scope 1 sources would a campus typically need to report?"
A: {"source": "both", "csv_source": "csv_table4", "query_type": "lookup"}

Q: "I'm having difficulties passing the final test about Sustainability, any tips and tricks?"
A: {"source": "none", "csv_source": null, "query_type": "lookup"}

Q: "What is the average GPA of students at Universitas Indonesia?"
A: {"source": "none", "csv_source": null, "query_type": "lookup"}

Q: "How does the UI GreenMetric scoring system compare to QS Stars and THE Impact rankings methodology?"
A: {"source": "none", "csv_source": null, "query_type": "lookup"}

=== OUTPUT FORMAT ===

{"source": "pdf", "csv_source": null, "query_type": "lookup"}
{"source": "csv", "csv_source": "csv_appendix1", "query_type": "aggregate"}
{"source": "both", "csv_source": "csv_appendix1", "query_type": "lookup"}

Respond with ONLY a valid JSON object. No markdown, no explanation, no code fences.

=== RULES ===

- "pdf" — answerable from the narrative guidelines alone (no tabular data needed).

- "csv" — answerable from structured tabular data alone. csv_source MUST be set to the single most relevant CSV source.

- "both" — use ONLY when the question explicitly references or directly connects a concept from the guidelines narrative to a specific CSV table. csv_source MUST be set.

- "none" — the information is not present in any of the sources. csv_source must be null. This includes questions about other universities' internal data, other ranking systems, general life advice, or information outside UI GreenMetric.

- "lookup" — the answer is a single fact, value, definition, or description from one or a few rows/documents. Use for specific indicators, single-row matches, or focused explanation. This is the default.

- "aggregate" — the question requires counting, listing all items, finding the maximum/minimum, comparing across multiple rows, or summarising a whole category. Use ONLY for queries that need the retriever to see the full set, not a sample. Examples: "how many questions...", "which category has the most...", "list all coordinators from...", "sebutkan seluruh...", "apa saja semua...".

- Set csv_source to null for "pdf" and "none" only.

- query_type must ALWAYS be set — "lookup" is the correct default for most queries, including all "pdf" and "none" queries.

- Questions may be in English, Indonesian, or mixed between both. Route based on content, not language."""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route(query: str,*,conversation_history: list[dict] | None = None) -> dict:
    """Classify *query* into a data-source category for retrieval.

    Uses an LLM call with few-shot examples embedded in
    ``ROUTER_SYSTEM_PROMPT`` to determine which ChromaDB source(s) the
    retriever should query.

    Parameters:
        query:                 The user's question.
        conversation_history:  Prior user/assistant message pairs to provide
                               context for follow-up questions.  Each dict
                               has ``"role"`` (``"user"`` or ``"assistant"``)
                               and ``"content"`` keys.  Optional.

    Returns:
        dict with keys:

        * ``"source"`` — one of ``"pdf"``, ``"csv"``, ``"both"``, ``"none"``.
        * ``"csv_source"`` — ``None`` when *source* is ``"pdf"`` or
          ``"none"``, otherwise a string identifying the specific CSV source
          (``"csv_appendix1"``, ``"csv_appendix2"``, ``"csv_appendix3"``,
          ``"csv_table1"``, ``"csv_table2"``, or ``"csv_table4"``).
        * ``"query_type"`` — ``"lookup"`` for single-fact queries (the
          default for most questions, including all ``"pdf"`` and
          ``"none"`` routes).  ``"aggregate"`` when the query requires
          counting, listing all items, finding a maximum/minimum, or
          comparing across multiple rows.  Used by the retriever to decide
          whether to attach parent chunks.
    """
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": query})

    response = ROUTER_CLIENT.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        temperature=0.0,
    )

    try:
        return json.loads(response.choices[0].message.content.strip())
    except (json.JSONDecodeError, KeyError):
        return {"source": "none", "csv_source": None, "query_type": "lookup"}