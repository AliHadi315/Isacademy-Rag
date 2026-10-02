"""Data preparation for statistics and PCA.

WHAT   Numeric-feature detection, missing-value handling, constant-column
       removal and standardisation.
WHY    PCA is scale-sensitive and cannot see NaNs; skipping this step is the
       classic way to produce a beautiful, meaningless projection.
LIB    pandas + scikit-learn (SimpleImputer, StandardScaler).
IN     pandas.DataFrame
OUT    PreparedData(matrix, features, n_dropped, notes)
NEXT   -> pca.run_pca / visualizations
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.utils.logging import get_logger

log = get_logger("preprocessing")

MIN_ROWS_FOR_PCA = 5
MIN_FEATURES_FOR_PCA = 2


class PreprocessingError(ValueError):
    pass


@dataclass
class PreparedData:
    matrix: object                      # numpy array, shape (n_samples, n_features)
    features: list = field(default_factory=list)
    n_samples: int = 0
    notes: list = field(default_factory=list)
    dropped: list = field(default_factory=list)


def numeric_features(df, min_unique: int = 2) -> list:
    """Columns that are genuinely numeric AND actually vary.

    An all-identical column contributes zero variance: it cannot help PCA and
    it makes the scaler divide by ~0.
    """
    import pandas as pd

    out = []
    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(float)
        clean = series.dropna()
        if len(clean) < 2 or clean.nunique() < min_unique:
            continue
        out.append(str(column))
    return out


def categorical_features(df) -> list:
    import pandas as pd

    return [
        str(c) for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= max(30, len(df) // 4)
    ]


def coerce_numeric(df):
    """Rescue numeric columns that arrived as text ('1,200', '45%', '(3)')."""
    import pandas as pd

    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(out[column]):
            continue
        cleaned = (
            out[column].astype(str)
            .str.strip()
            .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            .str.replace(r"[,\s$%]", "", regex=True)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            out[column] = numeric
    return out


def prepare_for_pca(df, features: list | None = None,
                    impute_strategy: str = "median") -> PreparedData:
    """Select -> impute -> drop constants -> standardise. Raises on unusable data."""
    import numpy as np

    notes: list = []
    dropped: list = []

    frame = coerce_numeric(df)
    chosen = [f for f in (features or numeric_features(frame)) if f in frame.columns]
    if len(chosen) < MIN_FEATURES_FOR_PCA:
        raise PreprocessingError(
            "PCA needs at least " + str(MIN_FEATURES_FOR_PCA) + " numeric features that vary; "
            "this dataset has " + str(len(chosen)) + "."
        )

    subset = frame[chosen].astype(float)

    # rows that are entirely missing carry no information
    before = len(subset)
    subset = subset.dropna(how="all")
    if len(subset) < before:
        notes.append("Dropped " + str(before - len(subset)) + " all-empty row(s).")

    if len(subset) < MIN_ROWS_FOR_PCA:
        raise PreprocessingError(
            "PCA needs at least " + str(MIN_ROWS_FOR_PCA) + " usable rows; this dataset has "
            + str(len(subset)) + "."
        )

    missing = int(subset.isna().sum().sum())
    if missing:
        try:
            from sklearn.impute import SimpleImputer

            values = SimpleImputer(strategy=impute_strategy).fit_transform(subset)
        except ImportError:
            values = subset.fillna(subset.median(numeric_only=True)).to_numpy()
        notes.append("Imputed " + str(missing) + " missing value(s) using the column " + impute_strategy + ".")
    else:
        values = subset.to_numpy()

    values = np.asarray(values, dtype=float)

    # constants can appear after imputation
    variances = values.var(axis=0)
    keep = variances > 1e-12
    if not keep.all():
        dropped = [c for c, k in zip(chosen, keep) if not k]
        notes.append("Dropped zero-variance feature(s): " + ", ".join(dropped) + ".")
        values = values[:, keep]
        chosen = [c for c, k in zip(chosen, keep) if k]

    if len(chosen) < MIN_FEATURES_FOR_PCA:
        raise PreprocessingError(
            "After removing constant columns fewer than "
            + str(MIN_FEATURES_FOR_PCA) + " features remain."
        )

    try:
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(values)
    except ImportError:
        centred = values - values.mean(axis=0)
        scaled = centred / np.clip(centred.std(axis=0), 1e-12, None)
    notes.append("Standardised " + str(len(chosen)) + " features (zero mean, unit variance).")

    log.info("prepared %d rows x %d features for PCA", scaled.shape[0], scaled.shape[1])
    return PreparedData(
        matrix=scaled, features=chosen, n_samples=int(scaled.shape[0]),
        notes=notes, dropped=dropped,
    )
