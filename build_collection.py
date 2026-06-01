"""Build (or rebuild) the ChromaDB collection from all 7 data sources.

Chunks every source, prints per-source counts + sanity check against the
expected 317-chunk total, then embeds with Qwen3 and persists into a
single ``greenmetric_v10`` collection.  Safe to re-run — overwrites
existing chunks with matching IDs.
"""

import sys
import pandas as pd
from src.chunker import (
    chunk_csv,
    chunk_markdown,
    _fmt_appendix1,
    _meta_appendix1,
    _fmt_appendix2,
    _meta_appendix2,
    _fmt_appendix3,
    _meta_appendix3,
    _fmt_table1,
    _meta_table1,
    _fmt_table2,
    _meta_table2,
    _fmt_table4,
    _meta_table4,
)

EXPECTED_TOTAL = 317  # from v0.5 README


def chunk_all() -> dict[str, list[dict]]:
    """Chunk every source.  Returns {source_name: [chunk_dicts]}."""

    sources: dict[str, list[dict]] = {}

    # --- PDF (markdown guidelines) ---
    sources["pdf"] = chunk_markdown("rag_data/guidelines_markdown.md")

    # --- csv_appendix1 ---
    df1 = pd.read_csv(
        "rag_data/appendix1_questionnairemasterandscoring.csv",
        sep=";", encoding="latin-1", dtype={"no": str},
    )
    df1["answer"] = df1["answer"].astype(str).str.replace("?", "\u2264", regex=False)
    sources["csv_appendix1"] = chunk_csv(
        df1, "no",
        source="csv_appendix1", chunk_type="question",
        format_fn=_fmt_appendix1, metadata_fn=_meta_appendix1,
    )

    # --- csv_appendix2 ---
    df2 = pd.read_csv("rag_data/appendix2_listofgreenbuildingelements.csv", sep=";")
    sources["csv_appendix2"] = chunk_csv(
        df2, "element_category",
        source="csv_appendix2", chunk_type="category",
        format_fn=_fmt_appendix2, metadata_fn=_meta_appendix2,
    )

    # --- csv_appendix3 ---
    df3 = pd.read_csv(
        "rag_data/appendix3_listanddescriptionofsmartbuildingrequirements.csv",
        sep="\t",
    )
    sources["csv_appendix3"] = chunk_csv(
        df3, "field_code",
        source="csv_appendix3", chunk_type="field_code",
        format_fn=_fmt_appendix3, metadata_fn=_meta_appendix3,
    )

    # --- csv_table1 ---
    df4 = pd.read_csv("rag_data/table1_nationalcoordinators.csv", sep=";")
    sources["csv_table1"] = chunk_csv(
        df4, "country",
        source="csv_table1", chunk_type="country",
        format_fn=_fmt_table1, metadata_fn=_meta_table1,
    )

    # --- csv_table2 ---
    df5 = pd.read_csv("rag_data/table2_categoriesusedandweighting.csv", sep=";")
    sources["csv_table2"] = chunk_csv(
        df5, "category",
        source="csv_table2", chunk_type="category",
        format_fn=_fmt_table2, metadata_fn=_meta_table2,
    )

    # --- csv_table4 ---
    df6 = pd.read_csv("rag_data/table4_greenhousegasemissionsources.csv", sep=";")
    sources["csv_table4"] = chunk_csv(
        df6, "scope",
        source="csv_table4", chunk_type="scope",
        format_fn=_fmt_table4, metadata_fn=_meta_table4,
    )

    return sources


def print_sanity(sources: dict[str, list[dict]]) -> int:
    """Print per-source chunk counts and compare total against expected."""
    print("=" * 55)
    print("  SANITY CHECK — chunk counts")
    print("=" * 55)
    total = 0
    for name in sources:
        n = len(sources[name])
        total += n
        print(f"  {name:<22} {n:>4} chunks")
    print(f"  {'─' * 28}")
    print(f"  {'TOTAL':<22} {total:>4} chunks  (expected: {EXPECTED_TOTAL})")
    if total == EXPECTED_TOTAL:
        print(f"  ✓ MATCH")
    else:
        delta = total - EXPECTED_TOTAL
        sign = "+" if delta > 0 else ""
        print(f"  ✗ DELTA: {sign}{delta} — investigate before embedding!")
        print()
        print("  Aborting. Fix the data or update EXPECTED_TOTAL and re-run.")
    print()
    return total


if __name__ == "__main__":
    # ---- Phase 1: chunk only (no model download) ----
    sources = chunk_all()
    total = print_sanity(sources)

    if total != EXPECTED_TOTAL:
        print("Chunk count mismatch — aborting before embedding.")
        print("Update EXPECTED_TOTAL if this is intentional, then re-run.")
        sys.exit(1)

    # ---- Phase 2: embed + store (BGE-M3 download happens here) ----
    from src.embedder import store

    print(f"Embedding {total} chunks with Qwen3 → chroma_db/greenmetric_qwen3 ...")
    store(sources, collection_name="greenmetric_qwen3")
    print("Done — collection greenmetric_qwen3 is ready.")
