"""Dataset PCA - on real numerical features (not embeddings).

    dataset -> numeric columns -> impute missing -> StandardScaler
            -> PCA(2) -> 2-D coordinates -> KMeans -> insights

This is a different thing from the semantic PCA in `semantic_pca.py`, which
projects embedding vectors. The two are deliberately never mixed.

Standardisation is not optional: PCA follows variance, so an unscaled column
measured in millions would drown one measured in percent.
"""
from __future__ import annotations

from app.analytics.clustering import cluster_vectors
from app.analytics.preprocessing import PreprocessingError, prepare_for_pca
from app.models.schemas import PCAResult
from app.utils.logging import get_logger

log = get_logger("pca")


def run_pca(df, features: list | None = None, n_components: int = 2,
            n_clusters: int = 3, dataset_name: str = "dataset") -> PCAResult:
    """Run PCA on a DataFrame. Never raises - unusable data returns a reason."""
    try:
        prepared = prepare_for_pca(df, features)
    except PreprocessingError as exc:
        log.info("PCA declined for %s: %s", dataset_name, exc)
        return PCAResult(performed=False, reason=str(exc))
    except Exception as exc:
        log.exception("PCA preprocessing failed")
        return PCAResult(performed=False, reason="Could not prepare the data: " + str(exc))

    n_components = max(1, min(n_components, len(prepared.features), prepared.n_samples))

    try:
        import numpy as np
        from sklearn.decomposition import PCA
    except ImportError as exc:
        return PCAResult(performed=False, reason="scikit-learn is not installed: " + str(exc))

    try:
        model = PCA(n_components=n_components, random_state=0)
        coordinates = model.fit_transform(prepared.matrix)
    except Exception as exc:
        log.exception("PCA fit failed")
        return PCAResult(performed=False, reason="PCA failed: " + str(exc))

    labels, _ = cluster_vectors(coordinates, n_clusters)

    result = PCAResult(
        performed=True,
        reason="; ".join(prepared.notes),
        components=n_components,
        features=prepared.features,
        explained_variance=[round(float(v), 6) for v in model.explained_variance_ratio_],
        cumulative_variance=[
            round(float(v), 6) for v in np.cumsum(model.explained_variance_ratio_)
        ],
        loadings={
            feature: [round(float(model.components_[c][i]), 6) for c in range(n_components)]
            for i, feature in enumerate(prepared.features)
        },
        coordinates=[[round(float(v), 6) for v in row] for row in coordinates],
        clusters=list(labels),
        n_samples=prepared.n_samples,
    )
    log.info(
        "PCA on %s: %d components retain %.1f%% of variance (%d x %d)",
        dataset_name, n_components, result.cumulative_variance[-1] * 100,
        result.n_samples, len(prepared.features),
    )
    return result


def top_loadings(result: PCAResult, component: int = 0, limit: int = 5) -> list:
    """The features that dominate one component, strongest |weight| first."""
    if not result.performed or component >= result.components:
        return []
    ranked = sorted(
        ((f, w[component]) for f, w in result.loadings.items()),
        key=lambda kv: -abs(kv[1]),
    )
    return [{"feature": f, "loading": round(w, 4)} for f, w in ranked[:limit]]


def describe(result: PCAResult, language: str = "en") -> list:
    """Plain-language notes, derived strictly from the numbers above."""
    if not result.performed:
        if language == "ar":
            return ["لم يتم تنفيذ PCA: " + result.reason]
        return ["PCA was not performed: " + result.reason]

    total = result.cumulative_variance[-1] * 100 if result.cumulative_variance else 0.0
    drivers = top_loadings(result, 0, 3)
    driver_text = ", ".join(d["feature"] + " (" + str(d["loading"]) + ")" for d in drivers)

    if language == "ar":
        lines = [
            "تم اختزال " + str(len(result.features)) + " خاصية رقمية إلى "
            + str(result.components) + " مكوّنات عبر " + str(result.n_samples) + " صفاً.",
        ]
        for i, ratio in enumerate(result.explained_variance):
            lines.append("المكوّن PC" + str(i + 1) + " يفسّر " + ("%.1f%%" % (ratio * 100))
                         + " من التباين.")
        lines.append("مجتمعةً تحتفظ بـ " + ("%.1f%%" % total) + " من التباين.")
        if total < 50:
            lines.append("أقل من 50%: الصورة ثنائية الأبعاد ملخص ضعيف لهذه البيانات.")
        if driver_text:
            lines.append("المكوّن PC1 تقوده أساساً: " + driver_text + ".")
        return lines

    lines = [
        "PCA reduced " + str(len(result.features)) + " standardised numeric features to "
        + str(result.components) + " components across " + str(result.n_samples) + " rows.",
    ]
    for i, ratio in enumerate(result.explained_variance):
        lines.append("PC" + str(i + 1) + " explains " + ("%.1f%%" % (ratio * 100))
                     + " of total variance.")
    lines.append("Together they retain " + ("%.1f%%" % total) + " of the variance.")
    if total < 50:
        lines.append(
            "Below 50% retained: the 2-D picture is a weak summary of this dataset, "
            "so distances in it should not be over-interpreted."
        )
    if driver_text:
        lines.append("PC1 is driven mainly by " + driver_text + ".")
    return lines
