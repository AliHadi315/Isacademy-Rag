"""Retrieval Agent - find the relevant chunks, or admit there are none.

    query -> query embedding -> Chroma search -> similarity threshold -> results

Returning nothing is a valid, important outcome: `sufficient=False` is what
stops the answer stage from writing something unsupported.
"""
from __future__ import annotations

from app.config import settings
from app.models.schemas import RetrievalResult
from app.rag.embeddings import get_embedding_provider
from app.rag.vector_store import VectorStoreError, check_fingerprint, get_vector_store
from app.utils.logging import get_logger

log = get_logger("agent.retrieval")


class RetrievalAgent:
    def __init__(self, top_k: int | None = None, threshold: float | None = None):
        self.top_k = top_k or settings.top_k
        self._threshold_override = threshold

    @property
    def threshold(self) -> float:
        if self._threshold_override is not None:
            return self._threshold_override
        scale = getattr(get_embedding_provider(), "threshold_scale", 1.0)
        return round(settings.similarity_threshold * scale, 4)

    # ------------------------------------------------------------------
    def run(self, query: str, top_k: int | None = None,
            document_ids: list | None = None) -> RetrievalResult:
        """Retrieve. Never raises - problems come back as a message."""
        query = (query or "").strip()
        threshold = self.threshold
        if not query:
            return RetrievalResult(threshold=threshold, message="Empty question.")

        try:
            store = get_vector_store()
            if store.count() == 0:
                return RetrievalResult(
                    query=query, threshold=threshold,
                    message="No documents are indexed yet.",
                )

            provider = get_embedding_provider()
            mismatch = check_fingerprint(provider)
            if mismatch:
                log.error(mismatch)
                return RetrievalResult(query=query, threshold=threshold, message=mismatch)

            k = top_k or self.top_k
            vector = provider.embed_query(query)
            # over-fetch so the threshold has something to cut into
            hits = store.query(vector, top_k=max(k * 3, k + 5), document_ids=document_ids)
        except VectorStoreError as exc:
            log.error("retrieval failed: %s", exc)
            return RetrievalResult(query=query, threshold=threshold, message=str(exc))
        except Exception as exc:
            log.exception("unexpected retrieval failure")
            return RetrievalResult(query=query, threshold=threshold,
                                   message="Search failed: " + str(exc))

        kept = [h for h in hits if h.score >= threshold][:k]
        log.info("%d/%d chunks above threshold %.2f", len(kept), len(hits), threshold)

        if not kept:
            return RetrievalResult(
                query=query, threshold=threshold,
                message=("Nothing scored above the similarity threshold of "
                         + ("%.2f" % threshold) + "."),
            )
        return RetrievalResult(
            query=query, results=kept, threshold=threshold, sufficient=True,
            message=(str(len(kept)) + " passages from "
                     + str(len({r.document for r in kept})) + " document(s)."),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def context_block(results: list, max_chars: int = 9000) -> str:
        """Render retrieved chunks for a Gemini prompt, with source labels."""
        lines, used = [], 0
        for i, r in enumerate(results, 1):
            head = ("[" + str(i) + "] " + r.document + " | page " + str(r.page)
                    + " | similarity " + ("%.2f" % r.score))
            body = r.content.strip()
            if used + len(body) > max_chars:
                body = body[: max(0, max_chars - used)] + "..."
            lines.append(head + "\n" + body)
            used += len(body)
            if used >= max_chars:
                break
        return "\n\n".join(lines)
