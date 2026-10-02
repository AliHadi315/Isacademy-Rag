"""Table extraction from PDF pages.

WHAT   Uses PyMuPDF's table finder to lift tabular regions into row matrices +
       a Markdown rendering, and can promote a numeric table into a pandas
       DataFrame for the Data agent / PCA.
WHY    Numbers locked inside a PDF table are invisible to text RAG. Turning
       them into a DataFrame is what makes "run PCA on the table in the report"
       a real operation instead of a demo.
LIB    PyMuPDF `page.find_tables()`; pandas for the DataFrame promotion.
IN     a fitz.Page  /  ExtractedTable
OUT    list[ExtractedTable]  /  pandas.DataFrame
NEXT   -> chunker (as searchable markdown) and -> data_agent (as a DataFrame)
"""
from __future__ import annotations

import re

from app.models.schemas import ExtractedTable
from app.utils.logging import get_logger

log = get_logger("tables")

MAX_TABLES_PER_PAGE = 6


def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[(c or "").replace("\n", " ").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    header, body = norm[0], norm[1:]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def _looks_like_a_table(rows: list[list[str]]) -> bool:
    """Guard for the borderless pass: prose can otherwise masquerade as a table."""
    if len(rows) < 2 or len(rows[0]) < 2:
        return False
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        return False
    cells = [c for row in rows for c in row]
    filled = sum(1 for c in cells if c.strip())
    if filled / max(len(cells), 1) < 0.6:
        return False
    # real table cells are short; a paragraph split into columns is not
    return max(len(c) for c in cells) <= 120


def extract_page_tables(page, document_name: str, page_number: int) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    found, strict = [], True
    try:
        found = list(getattr(page.find_tables(), "tables", []))
        if not found:
            # PyMuPDF's default finder needs ruling lines. Plenty of real report
            # tables are borderless, so retry on text alignment - with a stricter
            # acceptance test, because this pass is much easier to fool.
            found = list(getattr(page.find_tables(strategy="text"), "tables", []))
            strict = False
    except Exception as exc:            # older PyMuPDF, or an odd page
        log.debug("find_tables unavailable on p%d: %s", page_number, exc)
        return tables

    safe_doc = re.sub(r"[^A-Za-z0-9_-]+", "_", document_name)[:60]
    for i, table in enumerate(found[:MAX_TABLES_PER_PAGE]):
        try:
            raw = table.extract()
            rows = [[("" if c is None else str(c)).strip() for c in row] for row in raw]
            rows = [r for r in rows if any(c for c in r)]
            if len(rows) < 2 or len(rows[0]) < 2:
                continue
            if not strict and not _looks_like_a_table(rows):
                continue
            tables.append(
                ExtractedTable(
                    table_id=safe_doc + "_p" + str(page_number) + "_t" + str(i),
                    document_name=document_name,
                    page_number=page_number,
                    rows=rows,
                    markdown=rows_to_markdown(rows),
                )
            )
        except Exception as exc:
            log.warning("table %d on p%d failed: %s", i, page_number, exc)
    return tables


def table_to_dataframe(table: ExtractedTable):
    """Promote a table to a DataFrame, coercing numeric-looking columns.

    Returns None when the table has no usable numeric content.
    """
    import pandas as pd

    if not table.rows or len(table.rows) < 2:
        return None
    header = [c.strip() or ("col_" + str(i)) for i, c in enumerate(table.rows[0])]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in table.rows[1:]]
    df = pd.DataFrame(body, columns=header)

    for col in df.columns:
        cleaned = (
            df[col].astype(str)
            .str.replace(r"[,\s$%]", "", regex=True)
            .str.replace(r"^\((.*)\)$", r"-\1", regex=True)     # (123) -> -123
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() >= 0.6:
            df[col] = numeric
    return df
