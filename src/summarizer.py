"""Aggregate query summarizer for the UI GreenMetric RAG system.

Extracts key statistics from raw `_fetch_all` chunks by sending them
to v4-flash with a targeted JSON extraction prompt. The summary
replaces the full chunk dump in the generator's context.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

_PROMPTS = {
    "csv_appendix1": """You are extracting aggregate statistics from UI GreenMetric indicator data.
Extract ONLY these fields from the criteria below. Output ONLY valid JSON.

Fields required:
- "category_counts": number of questions per category (SI, EC, WS, WR, TR, ED, GD)
- "total_indicators": sum of all category counts
- "max_score": highest max_score value among all criteria
- "min_score": lowest non-zero max_score value

=== CRITERIA DATA ===
""",
    "csv_table1": """You are extracting aggregate statistics from UI GreenMetric national coordinator data.
Output ONLY valid JSON.

Fields required:
- "total_countries": number of unique countries
- "total_universities": number of coordinator universities
- "coordinator_by_country": object mapping country name to list of university names

=== COORDINATOR DATA ===
""",
    "csv_table2": """You are extracting aggregate statistics from UI GreenMetric category weighting data.
Output ONLY valid JSON.

Fields required:
- "category_weights": object mapping category name (full name) to percentage value

=== CATEGORY DATA ===
""",
    "csv_table4": """You are extracting aggregate statistics from UI GreenMetric emission source data.
Output ONLY valid JSON.

Fields required:
- "emission_scopes": object mapping scope name to a brief one-line description

=== EMISSION DATA ===
""",
    "csv_appendix2": """You are extracting aggregate statistics from UI GreenMetric green building elements data.
Output ONLY valid JSON.

Fields required:
- "element_categories": list of element category names
- "total_new_construction_elements": total count of elements under new construction
- "total_existing_building_elements": total count of elements under existing building

=== GREEN BUILDING DATA ===
""",
    "csv_appendix3": """You are extracting aggregate statistics from UI GreenMetric smart building requirements data.
Output ONLY valid JSON.

Fields required:
- "field_codes": object mapping field code (B, S, E, A, I, L) to field name
- "total_requirements": total count of requirements across all fields

=== SMART BUILDING DATA ===
""",
    "pdf": """You are extracting key facts from UI GreenMetric guidelines.
Output ONLY valid JSON.

Fields required:
- "key_facts": list of the most important facts or statistics mentioned

=== GUIDELINES DATA ===
""",
}


def summarize_aggregate(chunks: list[dict], source: str = "csv_appendix1") -> str:
    """Extract structured aggregate facts from raw chunks via v4-flash.

    Parameters:
        chunks: List of chunk dicts from ``_fetch_all``.
        source: Source identifier (csv_appendix1, csv_table1, etc.) to
                select the appropriate extraction prompt.

    Returns:
        str: JSON summary string, or a concatenated dump of all chunk
        content if extraction fails.
    """
    content = "\n\n".join(c["content"] for c in chunks)
    prompt = _PROMPTS.get(source, _PROMPTS["csv_appendix1"]) + content

    try:
        response = _CLIENT.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        json.loads(raw)
        return raw
    except (json.JSONDecodeError, Exception):
        return content
