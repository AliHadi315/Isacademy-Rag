"""Structured-data discovery and profiling.

WHAT   Finds the tabular data available to the session - CSV/Excel files the
       user uploaded, plus numeric tables lifted out of indexed PDFs - loads
       one into a DataFrame, and profiles it.
WHY    "Analyse the data" is only answerable if the system knows what data
       exists.  Promoting PDF tables into DataFrames is what connects the RAG
       half of the project to the analytics half.
LIB    pandas (+ openpyxl for .xlsx).
IN     the registry / a dataset name / a DataFrame
OUT    DatasetProfile, and chart recommendations derived from the profile
NEXT   -> data_agent -> pca / visualizations
"""
from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.document_processing.metadata import get_registry
from app.models.schemas import DatasetProfile
from app.utils.logging import get_logger

from .preprocessing import categorical_features, coerce_numeric, numeric_features

log = get_logger("data_detector")

MAX_PROFILE_ROWS = 20000       # keep profiling snappy on large uploads


def available_datasets() -> dict:
    """{name: {"path", "source"}} - uploaded files plus PDF-derived tables."""
    registry = get_registry()
    found = dict(registry.datasets())

    for path in sorted(settings.datasets_dir.glob("*")):
        if path.suffix.lower() in settings.allowed_data_ext and path.name not in found:
            found[path.name] = {"path": str(path), "source": "uploaded"}
    return found


def load_dataset(name_or_path: str):
    """Load a dataset by registry name or path. Raises FileNotFoundError/ValueError."""
    import pandas as pd

    entry = available_datasets().get(name_or_path)
    path = Path(entry["path"]) if entry else Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError("Dataset not found: " + str(name_or_path))

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported dataset type: " + suffix)

    if df.empty:
        raise ValueError("'" + path.name + "' contains no rows.")
    log.info("dataset loaded: %s (%d rows x %d cols)", path.name, len(df), df.shape[1])
    return coerce_numeric(df)


def tables_from_documents(min_numeric_columns: int = 2) -> dict:
    """Numeric tables extracted from indexed PDFs, as loadable DataFrames.

    ponytail: re-extracts from the source PDF on demand instead of caching the
    frames - a table page parse is milliseconds and the alternative is a second
    persistence layer to keep in sync.
    """
    from app.document_processing.pdf_extractor import extract_pdf
    from app.document_processing.table_extractor import table_to_dataframe

    out: dict = {}
    for document in get_registry().all():
        if document.table_count == 0 or document.file_type != "pdf":
            continue
        path = Path(document.path)
        if not path.exists():
            continue
        try:
            pages, _ = extract_pdf(path, document.name, with_images=False, with_tables=True)
        except Exception as exc:
            log.warning("could not re-read tables from %s: %s", document.name, exc)
            continue
        for page in pages:
            for table in page.tables:
                df = table_to_dataframe(table)
                if df is None or len(numeric_features(df)) < min_numeric_columns:
                    continue
                label = document.name + " - table p" + str(page.page_number)
                out[label] = df
    return out


def profile_dataframe(df, name: str = "dataset", path: str = "",
                      source: str = "uploaded") -> DatasetProfile:
    frame = df.head(MAX_PROFILE_ROWS)
    numeric = numeric_features(frame)

    describe: dict = {}
    if numeric:
        stats = frame[numeric].describe().to_dict()
        describe = {
            column: {k: (None if v != v else round(float(v), 6)) for k, v in values.items()}
            for column, values in stats.items()
        }

    correlations: dict = {}
    top: list = []
    if len(numeric) >= 2:
        corr = frame[numeric].corr(numeric_only=True)
        correlations = {
            c: {r: (None if corr.loc[r, c] != corr.loc[r, c] else round(float(corr.loc[r, c]), 4))
                for r in corr.index}
            for c in corr.columns
        }
        seen = set()
        for a in corr.columns:
            for b in corr.index:
                if a == b or (b, a) in seen:
                    continue
                seen.add((a, b))
                value = corr.loc[b, a]
                if value == value:
                    top.append({"a": a, "b": b, "r": round(float(value), 4)})
        top.sort(key=lambda d: -abs(d["r"]))
        top = top[:8]

    profile = DatasetProfile(
        name=name,
        path=path,
        rows=int(len(df)),
        columns=int(df.shape[1]),
        column_names=[str(c) for c in df.columns],
        numeric_features=numeric,
        categorical_features=categorical_features(frame),
        missing_by_column={str(c): int(v) for c, v in df.isna().sum().items() if v},
        describe=describe,
        correlations=correlations,
        top_correlations=top,
        source=source,
    )
    log.info(
        "profiled %s: %d rows, %d cols, %d numeric",
        name, profile.rows, profile.columns, len(numeric),
    )
    return profile


def recommend_visualizations(profile: DatasetProfile) -> list:
    """Chart recommendations that follow from the profile, not from a list."""
    recommended: list = []
    n_numeric = len(profile.numeric_features)

    if n_numeric >= 2:
        recommended.append("correlation_matrix")
    if n_numeric >= 3 and profile.rows >= 5:
        recommended.append("pca")
    if n_numeric >= 2:
        recommended.append("scatter")
    if n_numeric >= 1:
        recommended.append("histogram")
    if profile.categorical_features and n_numeric >= 1:
        recommended.append("bar")
    if n_numeric >= 2 and profile.rows >= 8:
        recommended.append("line")
    if n_numeric >= 3 and profile.rows >= 5:
        recommended.append("heatmap")
    if profile.missing_by_column:
        recommended.append("missing")
    return recommended


def observations(profile: DatasetProfile) -> list:
    """Facts computed from the profile - no interpretation, no invention."""
    notes = [
        str(profile.rows) + " rows x " + str(profile.columns) + " columns; "
        + str(len(profile.numeric_features)) + " numeric, "
        + str(len(profile.categorical_features)) + " categorical.",
    ]

    missing_total = sum(profile.missing_by_column.values())
    if missing_total:
        worst = max(profile.missing_by_column.items(), key=lambda kv: kv[1])
        notes.append(
            str(missing_total) + " missing value(s); worst column '" + worst[0] + "' with "
            + str(worst[1]) + " (" + ("%.1f%%" % (100 * worst[1] / max(profile.rows, 1))) + ")."
        )
    else:
        notes.append("No missing values.")

    for pair in profile.top_correlations[:3]:
        strength = (
            "strong" if abs(pair["r"]) >= 0.7 else "moderate" if abs(pair["r"]) >= 0.4 else "weak"
        )
        direction = "positive" if pair["r"] > 0 else "negative"
        notes.append(
            strength + " " + direction + " correlation between '" + pair["a"] + "' and '"
            + pair["b"] + "' (r = " + str(pair["r"]) + ")."
        )

    for column, stats in list(profile.describe.items())[:3]:
        mean, std = stats.get("mean"), stats.get("std")
        if mean is not None and std:
            notes.append(
                "'" + column + "': mean " + ("%.4g" % mean) + ", sd " + ("%.4g" % std)
                + ", range " + ("%.4g" % stats.get("min", 0)) + " to "
                + ("%.4g" % stats.get("max", 0)) + "."
            )
    return notes
