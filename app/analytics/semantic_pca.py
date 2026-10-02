"""Semantic PCA - where the question sits among the documents.

This runs on EVERY question that has enough neighbours, and answers six things
explicitly:

  1. Where is the query?          -> the query point in the projection
  2. Which cluster is it in?      -> query_cluster
  3. Which items are closest?     -> closest[]
  4. What are their scores?       -> real cosine similarity on full vectors
  5. Why are they close?          -> explanation (written by Gemini from the
                                     actual retrieved text)
  6. What else is nearby?         -> nearby_other_cluster[]

Both the query and the documents are projected by the SAME fitted PCA, and the
query vector is the same one retrieval used - so the picture and the ranking
tell a consistent story.
"""
from __future__ import annotations

from app.analytics.clustering import assign_cluster, cluster_terms, cluster_vectors
from app.analytics.similarity import cosine_to_query
from app.config import settings
from app.models.gemini import GeminiError, get_gemini
from app.models.schemas import PCAPoint, SemanticPCAResult
from app.utils.helpers import truncate
from app.utils.logging import get_logger

log = get_logger("semantic_pca")

MIN_POINTS = 4          # below this a 2-D projection says nothing

EXPLAIN_PROMPT = {
    "en": """The user asked this question:
{question}

These document passages are the ones semantically closest to it, with their
cosine similarity scores:

{items}

Recurring terms in this group: {terms}

In 2-4 sentences, explain WHY these passages are semantically close to the
question. Refer to the actual topics they share. Do not invent content that is
not in the passages above, and do not mention embeddings, vectors or PCA.""",
    "ar": """طرح المستخدم هذا السؤال:
{question}

هذه المقاطع من المستندات هي الأقرب دلالياً إليه، مع درجات التشابه:

{items}

المصطلحات المتكررة في هذه المجموعة: {terms}

في جملتين إلى أربع جمل، اشرح لماذا تُعد هذه المقاطع قريبة دلالياً من السؤال.
أشر إلى المواضيع المشتركة الفعلية. لا تخترع محتوى غير موجود أعلاه، ولا تذكر
التضمينات أو المتجهات أو PCA.""",
}


def build_semantic_pca(query: str, query_vector: list, language: str = "en",
                       document_ids: list | None = None,
                       explain: bool = True) -> SemanticPCAResult:
    """Project the query and its neighbourhood. Never raises."""
    from app.rag.vector_store import get_vector_store

    try:
        store = get_vector_store()
        if store.count() == 0:
            return SemanticPCAResult(reason="empty_index")

        size = min(settings.pca_neighbors, settings.pca_max_points)
        results, vectors = store.neighborhood(query_vector, size, document_ids)
    except Exception as exc:
        log.warning("semantic PCA neighbourhood failed: %s", exc)
        return SemanticPCAResult(reason=str(exc))

    if len(results) < MIN_POINTS or len(vectors) != len(results):
        return SemanticPCAResult(reason="not_enough_points")

    try:
        import numpy as np
        from sklearn.decomposition import PCA
    except ImportError as exc:
        return SemanticPCAResult(reason="scikit-learn is not installed: " + str(exc))

    # --- similarity on the FULL vectors, not the projection ---
    similarities = cosine_to_query(query_vector, vectors)

    # --- clustering on the FULL vectors ---
    labels, model = cluster_vectors(vectors, settings.pca_clusters)
    query_cluster = assign_cluster(model, query_vector)
    terms = cluster_terms([r.content for r in results], labels)

    # --- project documents + query with one fitted PCA ---
    try:
        matrix = np.asarray(vectors, dtype="float32")
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(matrix)
        query_xy = pca.transform(np.asarray(query_vector, dtype="float32").reshape(1, -1))[0]
        explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    except Exception as exc:
        log.warning("PCA projection failed: %s", exc)
        return SemanticPCAResult(reason=str(exc))

    points = [
        PCAPoint(
            x=round(float(coords[i][0]), 4),
            y=round(float(coords[i][1]), 4),
            label=results[i].document + " p." + str(results[i].page),
            document=results[i].document,
            page=results[i].page,
            cluster=int(labels[i]),
            similarity=similarities[i],
            snippet=truncate(results[i].content, 160),
        )
        for i in range(len(results))
    ]
    query_point = PCAPoint(
        x=round(float(query_xy[0]), 4), y=round(float(query_xy[1]), 4),
        label="Your question", cluster=query_cluster, similarity=1.0,
        is_query=True, snippet=truncate(query, 160),
    )

    ranked = sorted(points, key=lambda p: -p.similarity)
    closest = ranked[:5]

    # "which cluster is closest?" is a different question from "which cluster
    # is the query in?" - KMeans assigns the query by centroid distance, while
    # this is a vote among the passages that actually scored highest.
    votes: dict = {}
    for point in closest:
        votes[point.cluster] = votes.get(point.cluster, 0.0) + point.similarity
    closest_cluster = max(votes, key=votes.get) if votes else query_cluster

    nearby_other = [p for p in ranked if p.cluster != closest_cluster][:3]

    result = SemanticPCAResult(
        available=True,
        points=points + [query_point],
        query_cluster=query_cluster,
        closest_cluster=closest_cluster,
        n_clusters=len(set(labels)),
        explained_variance=explained,
        closest=closest,
        nearby_other_cluster=nearby_other,
        cluster_terms={int(k): v for k, v in terms.items()},
    )

    if explain:
        result.explanation = _explain(query, closest, terms.get(query_cluster, []), language)

    log.info(
        "semantic PCA: %d points, %d clusters, query in cluster %d, closest cluster %d, "
        "best similarity %.3f",
        len(points), result.n_clusters, query_cluster, closest_cluster,
        closest[0].similarity if closest else 0.0,
    )
    return result


def _explain(query: str, closest: list, terms: list, language: str) -> str:
    """Ask Gemini why these passages are close - grounded in their real text."""
    if not closest:
        return ""
    items = "\n".join(
        "- " + p.document + " (page " + str(p.page) + ", similarity "
        + ("%.2f" % p.similarity) + "): " + p.snippet
        for p in closest
    )
    template = EXPLAIN_PROMPT.get(language, EXPLAIN_PROMPT["en"])
    prompt = template.format(
        question=query, items=items, terms=", ".join(terms) or "-",
    )
    try:
        return get_gemini().generate(prompt, language=language)
    except GeminiError as exc:
        log.warning("cluster explanation unavailable: %s", exc)
        # a factual, non-invented fallback so the panel is never empty
        shared = ", ".join(terms[:5])
        if language == "ar":
            return (
                "المقاطع الأقرب تتشارك مصطلحات متكررة مثل: " + (shared or "—")
                + ". (تعذّر توليد شرح تفصيلي: " + str(exc) + ")"
            )
        return (
            "The closest passages share recurring terms such as: " + (shared or "—")
            + ". (A fuller explanation was unavailable: " + str(exc) + ")"
        )
