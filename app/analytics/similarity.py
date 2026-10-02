"""Cosine similarity helpers.

Similarity is always computed on the ORIGINAL embedding vectors, never on the
2-D PCA coordinates. PCA throws away most of the variance; distances in the
projection are an illustration, not a measurement.
"""
from __future__ import annotations


def cosine_to_query(query_vector, document_vectors) -> list:
    """Cosine similarity of one query vector against many document vectors."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if not document_vectors:
        return []
    q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    d = np.asarray(document_vectors, dtype="float32")
    return [round(float(v), 4) for v in cosine_similarity(q, d)[0]]


def rank_by_similarity(items: list, scores: list, limit: int = 5) -> list:
    """Pair items with scores and return the strongest `limit`, descending."""
    paired = sorted(zip(items, scores), key=lambda t: -t[1])
    return paired[:limit]
