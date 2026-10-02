"""Interactive Plotly figures for the UI.

Two families:
  * `semantic_map` - the question among its neighbouring chunks.
  * dataset charts  - PCA scatter, correlation heatmap, histogram, scatter, bar.

Matplotlib is still used for report PNGs (`visualizations.py`); Plotly is for
what the user interacts with on screen.
"""
from __future__ import annotations

from app.i18n import t
from app.models.schemas import SemanticPCAResult

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
QUERY_COLOR = "#dc2626"


def _layout(fig, title: str, x: str, y: str, rtl: bool = False):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title=x,
        yaxis_title=y,
        margin=dict(l=40, r=20, t=50, b=40),
        height=480,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        template="plotly_white",
    )
    return fig


def semantic_map(pca: SemanticPCAResult, lang: str = "en"):
    """The semantic PCA scatter: chunks by cluster, plus the query as a star."""
    import plotly.graph_objects as go

    fig = go.Figure()
    documents = [p for p in pca.points if not p.is_query]
    query = next((p for p in pca.points if p.is_query), None)

    for cluster in sorted({p.cluster for p in documents}):
        members = [p for p in documents if p.cluster == cluster]
        terms = ", ".join(pca.cluster_terms.get(cluster, [])[:3])
        name = t("cluster", lang) + " " + str(cluster) + (" — " + terms if terms else "")
        fig.add_trace(go.Scatter(
            x=[p.x for p in members],
            y=[p.y for p in members],
            mode="markers",
            name=name,
            marker=dict(size=11, color=PALETTE[cluster % len(PALETTE)],
                        line=dict(width=1, color="white"), opacity=0.85),
            customdata=[
                [p.document, p.page, p.cluster, p.similarity, p.snippet] for p in members
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + t("page", lang) + ": %{customdata[1]}<br>"
                + t("cluster", lang) + ": %{customdata[2]}<br>"
                + t("similarity", lang) + ": %{customdata[3]:.3f}<br>"
                "<i>%{customdata[4]}</i><extra></extra>"
            ),
        ))

    if query is not None:
        fig.add_trace(go.Scatter(
            x=[query.x], y=[query.y],
            mode="markers+text",
            name=t("query", lang),
            text=[t("query", lang)],
            textposition="top center",
            textfont=dict(size=13, color=QUERY_COLOR),
            marker=dict(size=22, color=QUERY_COLOR, symbol="star",
                        line=dict(width=2, color="white")),
            hovertemplate="<b>" + t("query", lang) + "</b><br>"
                          + t("cluster", lang) + ": " + str(pca.query_cluster)
                          + "<extra></extra>",
        ))

    ev = pca.explained_variance
    return _layout(
        fig, t("semantic_pca_map", lang),
        "PC1 (" + ("%.1f%%" % (ev[0] * 100 if ev else 0)) + ")",
        "PC2 (" + ("%.1f%%" % (ev[1] * 100 if len(ev) > 1 else 0)) + ")",
    )


def dataset_pca_scatter(pca, labels: list | None = None, lang: str = "en"):
    """Dataset PCA projection, coloured by cluster (or by a chosen column)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    xs = [row[0] for row in pca.coordinates]
    ys = [row[1] if len(row) > 1 else 0.0 for row in pca.coordinates]
    groups = labels if labels and len(labels) == len(xs) else pca.clusters

    if groups:
        for value in sorted(set(groups), key=str):
            idx = [i for i, g in enumerate(groups) if g == value]
            colour = PALETTE[sorted(set(groups), key=str).index(value) % len(PALETTE)]
            fig.add_trace(go.Scatter(
                x=[xs[i] for i in idx], y=[ys[i] for i in idx],
                mode="markers", name=str(value),
                marker=dict(size=9, color=colour, opacity=0.85,
                            line=dict(width=0.8, color="white")),
                hovertemplate="PC1 %{x:.3f}<br>PC2 %{y:.3f}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                                 marker=dict(size=9, color=PALETTE[0])))

    ev = pca.explained_variance
    return _layout(
        fig, "PCA",
        "PC1 (" + ("%.1f%%" % (ev[0] * 100 if ev else 0)) + ")",
        "PC2 (" + ("%.1f%%" % (ev[1] * 100 if len(ev) > 1 else 0)) + ")",
    )


def scree(pca, lang: str = "en"):
    import plotly.graph_objects as go

    labels = ["PC" + str(i + 1) for i in range(len(pca.explained_variance))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[v * 100 for v in pca.explained_variance],
                         name=t("explained_variance", lang), marker_color=PALETTE[0]))
    fig.add_trace(go.Scatter(x=labels, y=[v * 100 for v in pca.cumulative_variance],
                             name="Cumulative", mode="lines+markers",
                             line=dict(color=PALETTE[1], width=3)))
    return _layout(fig, t("explained_variance", lang), "", "%")


def correlation_heatmap(df, lang: str = "en"):
    import plotly.graph_objects as go

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("A correlation matrix needs at least 2 numeric columns.")
    corr = numeric.corr(numeric_only=True)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
    ))
    return _layout(fig, "Correlation", "", "")


def histogram(df, column: str, lang: str = "en"):
    import plotly.express as px

    fig = px.histogram(df, x=column, nbins=25, color_discrete_sequence=[PALETTE[0]])
    return _layout(fig, str(column), str(column), "Count")


def scatter(df, x: str, y: str, color: str | None = None, lang: str = "en"):
    import plotly.express as px

    fig = px.scatter(df, x=x, y=y, color=color, trendline=None,
                     color_discrete_sequence=PALETTE)
    return _layout(fig, str(y) + " / " + str(x), str(x), str(y))


def bar(df, category: str, value: str | None = None, lang: str = "en"):
    import plotly.express as px

    if value:
        data = df.groupby(category, as_index=False)[value].mean().head(20)
        fig = px.bar(data, x=category, y=value, color_discrete_sequence=PALETTE)
        return _layout(fig, str(value) + " / " + str(category), str(category), str(value))

    counts = df[category].value_counts().head(20).reset_index()
    counts.columns = [category, "count"]
    fig = px.bar(counts, x=category, y="count", color_discrete_sequence=PALETTE)
    return _layout(fig, str(category), str(category), "Count")
