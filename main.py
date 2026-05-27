import pandas as pd
from src.chunker import (
    chunk_csv,
    _fmt_appendix1,
    _meta_appendix1,
)
# 1. Load + pre-process
df = pd.read_csv(
    "rag_data/appendix1_questionnairemasterandscoring.csv",
    sep=";",
    encoding="latin-1",
    dtype={"no": str},
)
df["answer"] = df["answer"].astype(str).str.replace("?", "≤", regex=False)
# 2. Chunk

chunks = chunk_csv(
    df,
    "no",
    source="csv_appendix1",
    chunk_type="question",
    format_fn=_fmt_appendix1,
    metadata_fn=_meta_appendix1,
)