"""Visualization engine.

WHAT   Every chart the system can draw: PCA projection, scree, loadings,
       scatter, bar, line, histogram, correlation matrix, heatmap, missing-data
       map.  All render to PNG under data/visualizations/.
WHY    One place that owns figure style and file naming means the Streamlit
       page, the download buttons and the PDF report all show the same chart -
       no re-plotting, no drift.
LIB    matplotlib (Agg backend - no display server needed) + pandas.
IN     DataFrame / PCAResult
OUT    absolute path to a saved PNG
NEXT   -> Visualizations page, report generator
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # must precede pyplot; Streamlit has no GUI thread
import matplotlib.pyplot as plt  # noqa: E402

from app.config import settings  # noqa: E402
from app.models.schemas import PCAResult  # noqa: E402
from app.utils.helpers import timestamp_slug  # noqa: E402
from app.utils.logging import get_logger  # noqa: E402

log = get_logger("viz")

DPI = 140
FIGSIZE = (8, 5)
ACCENT = "#2563eb"
ACCENT_2 = "#f59e0b"
GRID = {"alpha": 0.25, "linestyle": "--", "linewidth": 0.7}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text))[:50].strip("_") or "figure"


def _new_path(kind: str, name: str) -> Path:
    return settings.viz_dir / (_slug(kind) + "_" + _slug(name) + "_" + timestamp_slug() + ".png")


def _finish(fig, path: Path, title: str) -> str:
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("figure saved: %s", path.name)
    return str(path)


def _style(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def plot_pca_projection(result: PCAResult, dataset_name: str = "dataset",
                        labels: list | None = None, label_name: str | None = None) -> str:
    if not result.performed or not result.coordinates:
        raise ValueError("Cannot plot a PCA that was not performed.")

    xs = [row[0] for row in result.coordinates]
    ys = [row[1] if len(row) > 1 else 0.0 for row in result.coordinates]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if labels and len(labels) == len(xs) and len(set(labels)) <= 12:
        groups = sorted(set(labels))
        colours = plt.cm.tab10(range(len(groups)))
        for colour, group in zip(colours, groups):
            idx = [i for i, label in enumerate(labels) if label == group]
            ax.scatter([xs[i] for i in idx], [ys[i] for i in idx],
                       s=45, alpha=0.8, color=colour, edgecolors="white",
                       linewidths=0.6, label=str(group))
        ax.legend(title=label_name or "group", fontsize=8, frameon=False)
    else:
        ax.scatter(xs, ys, s=45, alpha=0.8, color=ACCENT, edgecolors="white", linewidths=0.6)

    ev = result.explained_variance
    _style(
        ax,
        "PC1 - " + ("%.1f%%" % (ev[0] * 100 if ev else 0)) + " of variance",
        "PC2 - " + ("%.1f%%" % (ev[1] * 100 if len(ev) > 1 else 0)) + " of variance",
    )
    ax.axhline(0, color="#94a3b8", linewidth=0.8)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)

    total = result.cumulative_variance[-1] * 100 if result.cumulative_variance else 0
    ax.set_title(
        str(result.n_samples) + " samples - " + str(len(result.features))
        + " features - " + ("%.1f%%" % total) + " variance retained",
        fontsize=9, color="#475569",
    )
    return _finish(fig, _new_path("pca", dataset_name), "PCA projection - " + dataset_name)


def plot_pca_scree(result: PCAResult, dataset_name: str = "dataset") -> str:
    if not result.performed:
        raise ValueError("Cannot plot a PCA that was not performed.")
    ev = [v * 100 for v in result.explained_variance]
    cum = [v * 100 for v in result.cumulative_variance]
    labels = ["PC" + str(i + 1) for i in range(len(ev))]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(labels, ev, color=ACCENT, alpha=0.85, label="Individual")
    ax.plot(labels, cum, color=ACCENT_2, marker="o", linewidth=2, label="Cumulative")
    for i, value in enumerate(ev):
        ax.text(i, value + 1.5, "%.1f%%" % value, ha="center", fontsize=8)
    _style(ax, "Principal component", "Explained variance (%)")
    ax.set_ylim(0, max(105, max(cum + [0]) + 8))
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, _new_path("scree", dataset_name), "Explained variance - " + dataset_name)


def plot_pca_loadings(result: PCAResult, dataset_name: str = "dataset", limit: int = 12) -> str:
    if not result.performed or not result.loadings:
        raise ValueError("Cannot plot loadings for a PCA that was not performed.")
    ranked = sorted(result.loadings.items(), key=lambda kv: -abs(kv[1][0]))[:limit]
    names = [f for f, _ in ranked][::-1]
    pc1 = [w[0] for _, w in ranked][::-1]

    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.42 * len(names) + 1.4)))
    ax.barh(names, pc1, color=[ACCENT if v >= 0 else "#ef4444" for v in pc1], alpha=0.85)
    ax.axvline(0, color="#334155", linewidth=0.9)
    _style(ax, "Loading on PC1", "")
    return _finish(fig, _new_path("loadings", dataset_name), "PC1 loadings - " + dataset_name)


# ---------------------------------------------------------------------------
# General charts
# ---------------------------------------------------------------------------
def plot_correlation_matrix(df, name: str = "dataset", columns: list | None = None) -> str:
    import pandas as pd

    numeric = df[columns] if columns else df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("A correlation matrix needs at least 2 numeric columns.")
    corr = numeric.corr(numeric_only=True)

    size = max(5.0, 0.55 * len(corr) + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=8)
    if len(corr) <= 12:
        for i in range(len(corr)):
            for j in range(len(corr)):
                value = corr.values[i, j]
                ax.text(j, i, "%.2f" % value, ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.55 else "#1e293b")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    ax.grid(False)
    return _finish(fig, _new_path("correlation", name), "Correlation matrix - " + name)


def plot_heatmap(df, name: str = "dataset", max_rows: int = 40) -> str:
    """Standardised value heatmap - shows structure and outliers row by row."""
    numeric = df.select_dtypes(include="number").head(max_rows)
    if numeric.empty:
        raise ValueError("No numeric columns to draw a heatmap from.")
    values = numeric.to_numpy(dtype=float)
    centred = values - values.mean(axis=0)
    scaled = centred / (centred.std(axis=0) + 1e-12)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * numeric.shape[1] + 3), max(4.0, 0.22 * len(numeric) + 2)))
    image = ax.imshow(scaled, cmap="viridis", aspect="auto")
    ax.set_xticks(range(numeric.shape[1]))
    ax.set_xticklabels(numeric.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Row index")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Standard deviations from the mean")
    ax.grid(False)
    return _finish(fig, _new_path("heatmap", name), "Standardised value heatmap - " + name)


def plot_histogram(df, column: str, name: str = "dataset", bins: int = 25) -> str:
    series = df[column].dropna()
    if series.empty:
        raise ValueError("Column '" + str(column) + "' has no values to plot.")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(series, bins=min(bins, max(5, series.nunique())), color=ACCENT,
            alpha=0.85, edgecolor="white")
    mean = float(series.mean())
    ax.axvline(mean, color=ACCENT_2, linewidth=2, label="mean = %.3g" % mean)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, str(column), "Frequency")
    return _finish(fig, _new_path("hist", name + "_" + str(column)),
                   "Distribution of " + str(column))


def plot_scatter(df, x: str, y: str, name: str = "dataset", color_by: str | None = None) -> str:
    data = df[[x, y] + ([color_by] if color_by and color_by in df.columns else [])].dropna()
    if data.empty:
        raise ValueError("No overlapping values for " + str(x) + " and " + str(y) + ".")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if color_by and color_by in data.columns and data[color_by].nunique() <= 12:
        for colour, (group, part) in zip(
            plt.cm.tab10(range(data[color_by].nunique())), data.groupby(color_by)
        ):
            ax.scatter(part[x], part[y], s=42, alpha=0.8, color=colour,
                       edgecolors="white", linewidths=0.6, label=str(group))
        ax.legend(title=str(color_by), fontsize=8, frameon=False)
    else:
        ax.scatter(data[x], data[y], s=42, alpha=0.8, color=ACCENT,
                   edgecolors="white", linewidths=0.6)

    if len(data) >= 3:
        try:                                   # least-squares trend line
            import numpy as np

            slope, intercept = np.polyfit(data[x].astype(float), data[y].astype(float), 1)
            xs = np.linspace(data[x].min(), data[x].max(), 50)
            r = float(data[[x, y]].corr().iloc[0, 1])
            ax.plot(xs, slope * xs + intercept, color=ACCENT_2, linewidth=1.8,
                    label="trend (r = %.2f)" % r)
            ax.legend(fontsize=8, frameon=False)
        except Exception:
            pass

    _style(ax, str(x), str(y))
    return _finish(fig, _new_path("scatter", name), str(y) + " vs " + str(x))


def plot_bar(df, category: str, value: str | None = None, name: str = "dataset",
             limit: int = 15) -> str:
    if value and value in df.columns:
        series = df.groupby(category)[value].mean().sort_values(ascending=False).head(limit)
        ylabel = "Mean " + str(value)
    else:
        series = df[category].value_counts().head(limit)
        ylabel = "Count"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar([str(i) for i in series.index], series.values, color=ACCENT, alpha=0.85)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    _style(ax, str(category), ylabel)
    return _finish(fig, _new_path("bar", name + "_" + str(category)),
                   ylabel + " by " + str(category))


def plot_line(df, y: str, x: str | None = None, name: str = "dataset") -> str:
    frame = df.dropna(subset=[y])
    xs = frame[x] if x and x in frame.columns else range(len(frame))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, frame[y], color=ACCENT, linewidth=1.9, marker="o", markersize=3)
    _style(ax, str(x) if x else "Index", str(y))
    return _finish(fig, _new_path("line", name + "_" + str(y)), "Trend of " + str(y))


def plot_missing(df, name: str = "dataset") -> str:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        raise ValueError("This dataset has no missing values to chart.")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh([str(i) for i in missing.index][::-1], missing.values[::-1], color="#ef4444", alpha=0.85)
    _style(ax, "Missing values", "")
    return _finish(fig, _new_path("missing", name), "Missing values - " + name)


CHART_BUILDERS = {
    "correlation_matrix": plot_correlation_matrix,
    "heatmap": plot_heatmap,
    "histogram": plot_histogram,
    "scatter": plot_scatter,
    "bar": plot_bar,
    "line": plot_line,
    "missing": plot_missing,
}
