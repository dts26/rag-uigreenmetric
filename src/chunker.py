"""Chunkers for the UI GreenMetric RAG system.

Converts raw data sources (markdown, CSV tables) into unified
{content, metadata} dicts ready for embedding and ChromaDB storage.
"""

import pandas as pd
import re
from collections.abc import Callable


# ---------------------------------------------------------------------------
# Generic CSV chunker
# ---------------------------------------------------------------------------

def chunk_csv(
    df: pd.DataFrame,
    group_col: str,
    *,
    source: str,
    chunk_type: str,
    format_fn: Callable[[pd.DataFrame], str],
    metadata_fn: Callable[[pd.DataFrame], dict],
) -> list[dict]:
    """Generic group-based CSV chunker.

    Groups *df* by *group_col*, then delegates text formatting and
    metadata extraction to callables.  *source* and *chunk_type* are
    injected into the metadata dict automatically.

    Parameters:
        df:          Pre-processed DataFrame (encoding fixes, character
                     replacements already applied by the caller).
        group_col:   Column to group by (e.g. ``'no'``, ``'country'``,
                     ``'scope'``).
        source:      Source identifier injected into every chunk's metadata
                     (e.g. ``'csv_appendix1'``, ``'csv_table4'``).
        chunk_type:  Chunk type injected into every chunk's metadata
                     (e.g. ``'question'``, ``'category'``, ``'reference'``).
        format_fn:   ``Callable[[pd.DataFrame], str]``
                     Receives one group at a time.  Returns the chunk text.
        metadata_fn: ``Callable[[pd.DataFrame], dict]``
                     Receives one group at a time.  Returns domain-specific
                     metadata keys (``category``, ``question_no``, ...).
                     ``source`` and ``chunk_type`` are added by this
                     function automatically.

    Returns:
        list[dict]: One chunk per group.  Each chunk has keys ``"content"``
        (str) and ``"metadata"`` (dict).
    """
    chunks: list[dict] = []
    for no, group in df.groupby(group_col, sort=False):
        chunks.append({
            "content": format_fn(group),
            "metadata": metadata_fn(group)
            | {"source": source, "chunk_type": chunk_type},
        })
    return chunks


# ---------------------------------------------------------------------------
# Per-source format / metadata helpers
# ---------------------------------------------------------------------------

# --- appendix1 ----------------------------------------------------------------

def _fmt_appendix1(group: pd.DataFrame) -> str:
    no = group.iloc[0]["no"]
    text_content = f"""Question {no} — {group.iloc[0]['criteria']}
Category: {group.iloc[0]['category']}
Evidence Required: {group.iloc[0]['evidence_required']}
"""
    indicator_code = group.iloc[0]["indicator_code"] if not pd.isna(group.iloc[0]["indicator_code"]) else "Not Available"
    max_score = group.iloc[0]["max_score"] if not pd.isna(group.iloc[0]["max_score"]) else "Not Available"
    colored = group.iloc[0]["colored"] if not pd.isna(group.iloc[0]["colored"]) else "Not Available"

    text_content += f"Indicator Code: {indicator_code}\n"
    text_content += f"Max Score: {max_score}\n"
    text_content += f"Colored: {colored}\n"
    text_content += "Options:\n"

    for options in group.itertuples():
        if not pd.isna(options.calculated_score):
            text_content += f"{options.answer} (Calculated score: {options.calculated_score})\n"
        else:
            text_content += f"{options.answer} (Calculated score: Not Available)\n"

    return text_content


def _meta_appendix1(group: pd.DataFrame) -> dict:
    max_score = float(group.iloc[0]["max_score"]) if not pd.isna(group.iloc[0]["max_score"]) else -1.0  # max_score uses -1.0 as sentinel for unscored/ungraded criteria (no real score is negative)
    colored = group.iloc[0]["colored"] if not pd.isna(group.iloc[0]["colored"]) else "Not Available"
    return {
        "category": group.iloc[0]["category"],
        "question_no": group.iloc[0]["no"],
        "evidence_required": group.iloc[0]["evidence_required"],
        "max_score": max_score,
        "colored": colored,
    }


# --- appendix2 ----------------------------------------------------------------

def _fmt_appendix2(group: pd.DataFrame) -> str:
    text_content = f"Category: {group.iloc[0]['element_category']}\n"
    text_content += "Existing building category:\n"
    for sub_categories, element in group.loc[:, ["gbi_non-residential_existing_building_category", "gbi_non-residential_existing_building_element"]].itertuples(index=False):
        if pd.isna(sub_categories) and pd.isna(element):
            continue
        sub_categories = sub_categories if not pd.isna(sub_categories) else "Not Available"
        element = element if not pd.isna(element) else "Not Available"
        text_content += f"{sub_categories} | {element}\n"

    text_content += "\nNew construction category:\n"
    for sub_categories, element in group.loc[:, ["gbi_non-residential_new_construction_(nrnc)_category", "gbi_non-residential_new_construction_(nrnc)_element"]].itertuples(index=False):
        if pd.isna(sub_categories) and pd.isna(element):
            continue
        sub_categories = sub_categories if not pd.isna(sub_categories) else "Not Available"
        element = element if not pd.isna(element) else "Not Available"
        text_content += f"{sub_categories} | {element}\n"

    return text_content


def _meta_appendix2(group: pd.DataFrame) -> dict:
    return {
        "element_category": group.iloc[0]["element_category"],
    }


# --- appendix3 ----------------------------------------------------------------

def _fmt_appendix3(group: pd.DataFrame) -> str:
    text_content = f"""
Field code: {group.iloc[0]['field_code']}
Field category: {group.iloc[0]['field_name']}
"""
    text_content += "Requirements:\n"
    for code, name, description in group.loc[:, ["requirement_code", "requirement_name", "description"]].itertuples(index=False):
        text_content += f"{code} | {name}: {description}\n"

    return text_content


def _meta_appendix3(group: pd.DataFrame) -> dict:
    return {
        "field_code": group.iloc[0]["field_code"],
        "field_name": group.iloc[0]["field_name"],
    }


# --- table1 -------------------------------------------------------------------

def _fmt_table1(group: pd.DataFrame) -> str:
    text_content = f"Country: {group.iloc[0]['country']}\n"

    text_content += "Universities:\n"
    for university in group["university"]:
        text_content += f"{university}\n"
    return text_content


def _meta_table1(group: pd.DataFrame) -> dict:
    return {
        "country": group.iloc[0]["country"],
    }


# --- table2 -------------------------------------------------------------------

def _fmt_table2(group: pd.DataFrame) -> str:
    text_content = f"""
Category: {group.iloc[0]['category']}
Weight(%): {group.iloc[0]['percentage_of_total_points_(%)']}
"""
    return text_content


def _meta_table2(group: pd.DataFrame) -> dict:
    return {
        "category": group.iloc[0]["category"],
    }


# --- table4 -------------------------------------------------------------------

def _fmt_table4(group: pd.DataFrame) -> str:
    text_content = f"Scope category: {group.iloc[0]['scope']}\n"
    text_content += "Emission source:\n"
    for source, desc in group.loc[:, ["emission_source", "description_or_examples"]].itertuples(index=False):
        text_content += f"{source}: {desc}\n"
    return text_content


def _meta_table4(group: pd.DataFrame) -> dict:
    return {
        "scope": group.iloc[0]["scope"],
    }


# ---------------------------------------------------------------------------
# Markdown (PDF) chunker
# ---------------------------------------------------------------------------

def chunk_markdown(filepath: str) -> list[dict]:
    """Split UI GreenMetric guidelines markdown into hierarchical chunks.

    Strategy: heading-level structural chunking.
      - ``##``  → chunk_type ``"intro"``,     category = None
      - ``###`` → chunk_type ``"category"``,   category from heading text.
                    Sub-sections (``### a.``, ``### b.``) are appended to
                    the current chunk rather than split.
      - ``####`` → chunk_type ``"question"``,  single indicator description.
      - ``#####`` → appended to the current chunk, never triggers a split.

    A chunk is finalised when the NEXT heading of equal or higher rank is
    encountered.  Trailing content after the last heading is also captured
    as the final chunk.

    Returns:
        list[dict]: Each chunk has keys ``"content"`` (str) and
        ``"metadata"`` (dict with ``"source"``, ``"chunk_type"``,
        ``"category"``).
    """
    with open(filepath, "r", encoding="utf-8") as file:
        markdown_file = file.read()

    current_content = []
    chunks = []
    current_type = None
    current_category = None

    for line in markdown_file.splitlines():
        if line.startswith("## "):
            if current_type is not None:
                chunks.append({
                    "content": "\n".join(current_content).strip(),
                    "metadata": {
                        "source": "pdf",
                        "chunk_type": current_type,
                        "category": current_category,
                    },
                })
            current_category = None
            current_type = "intro"
            current_content = [line]
        elif line.startswith("### "):
            if re.match(r"### [a-z]\.", line):  # Check if it's not a category of questionnaire (starts with lowercase after ###)
                current_content.append(line)
            else:
                if current_type is not None:
                    chunks.append({
                        "content": "\n".join(current_content).strip(),
                        "metadata": {
                            "source": "pdf",
                            "chunk_type": current_type,
                            "category": current_category,
                        },
                    })
                current_type = "category"
                current_category = line[4:].strip()
                current_content = [line]
        elif line.startswith("#### "):
            if current_type is not None:
                chunks.append({
                    "content": "\n".join(current_content).strip(),
                    "metadata": {
                        "source": "pdf",
                        "chunk_type": current_type,
                        "category": current_category,
                    },
                })
            current_type = "question"
            current_content = [line]
        elif line.startswith("##### "):
            current_content.append(line)
        else:
            current_content.append(line)

    chunks.append({
        "content": "\n".join(current_content).strip(),
        "metadata": {
            "source": "pdf",
            "chunk_type": current_type,
            "category": current_category,
        },
    })

    return chunks
