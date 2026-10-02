"""KMeans clustering for the semantic map and for dataset PCA.

Clusters are fitted on the FULL embedding vectors (not the 2-D projection) so
that "which cluster is my question in?" is a statement about meaning rather
than about where a dot happened to land on a scatter plot.
"""
from __future__ import annotations

from app.utils.helpers import STOPWORDS
from app.utils.logging import get_logger

log = get_logger("clustering")

MIN_POINTS_PER_CLUSTER = 3


def choose_k(n_points: int, requested: int = 3) -> int:
    """A sane cluster count: never more clusters than the data can support."""
    if n_points < 2 * MIN_POINTS_PER_CLUSTER:
        return 1
    return max(2, min(requested, n_points // MIN_POINTS_PER_CLUSTER))


def cluster_vectors(vectors, k: int = 3):
    """Fit KMeans. Returns (labels, model) or ([0...], None) when k == 1."""
    import numpy as np

    data = np.asarray(vectors, dtype="float32")
    if len(data) == 0:
        return [], None

    k = choose_k(len(data), k)
    if k <= 1:
        return [0] * len(data), None

    try:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(data)
        return [int(v) for v in model.labels_], model
    except Exception as exc:
        log.warning("clustering failed (%s) - treating everything as one cluster", exc)
        return [0] * len(data), None


def assign_cluster(model, vector) -> int:
    """Which cluster does the query belong to?"""
    if model is None:
        return 0
    import numpy as np

    try:
        return int(model.predict(np.asarray(vector, dtype="float32").reshape(1, -1))[0])
    except Exception:
        return 0


def cluster_terms(texts: list, labels: list, limit: int = 6) -> dict:
    """The words that characterise each cluster.

    A term scores highly when it is frequent inside its cluster and rare
    outside it - which is what makes the "why are they close?" explanation
    concrete instead of generic.
    """
    import re

    per_cluster: dict = {}
    for text, label in zip(texts, labels):
        words = [
            w for w in re.findall(r"\w{3,}", (text or "").lower(), flags=re.UNICODE)
            if w not in STOPWORDS and not w.isdigit()
        ]
        per_cluster.setdefault(int(label), []).append(words)

    global_counts: dict = {}
    for buckets in per_cluster.values():
        for words in buckets:
            for w in set(words):
                global_counts[w] = global_counts.get(w, 0) + 1

    out: dict = {}
    for label, buckets in per_cluster.items():
        counts: dict = {}
        for words in buckets:
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        scored = sorted(
            counts.items(),
            key=lambda kv: -(kv[1] / (1 + global_counts.get(kv[0], 1))),
        )
        out[label] = [w for w, _ in scored[:limit]]
    return out
