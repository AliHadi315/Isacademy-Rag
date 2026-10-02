"""Embeddings - local sentence-transformers (not an LLM provider).

Gemini handles all reasoning and vision; embeddings stay local so retrieval is
free, fast and offline. Vectors are L2-normalised, so a dot product is cosine
similarity - which is what both the retriever and the semantic PCA rely on.

Default: BAAI/bge-small-en-v1.5 (384-d). BGE is asymmetric - queries get an
instruction prefix, passages do not - and that prefix is applied here, once,
so no caller has to remember it.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.config import settings
from app.utils.logging import get_logger

log = get_logger("embeddings")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingProvider(ABC):
    name: str = "abstract"
    dimension: int = 0
    # Cosine scores are not comparable across embedding families, so each
    # provider says how to rescale SIMILARITY_THRESHOLD (which is expressed on
    # the transformer scale).
    threshold_scale: float = 1.0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]: ...

    def describe(self) -> dict:
        return {"provider": self.name, "dimension": self.dimension}


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model
        self.name = self.model_name
        log.info("loading embedding model %s", self.model_name)
        self.model = SentenceTransformer(self.model_name, device=device or settings.embedding_device)
        getter = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dimension = int(getter())
        self._is_bge = "bge" in self.model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts), batch_size=settings.embedding_batch_size,
            normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        text = (BGE_QUERY_PREFIX + query) if self._is_bge else query
        vector = self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )[0]
        return vector.tolist()


class HashingEmbeddingProvider(EmbeddingProvider):
    """Offline fallback: hashed bag-of-ngrams with sublinear term frequency.

    A real (if shallow) vector space - cosine over it is genuine lexical
    similarity, so retrieval and PCA still work. It matches words, not meaning.
    Used only when the transformer model cannot be loaded, and labelled as such
    wherever it appears in the UI.
    """

    name = "hashing-fallback"
    threshold_scale = 0.35

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
        bigrams = [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]
        return words + bigrams

    def _vector(self, text: str) -> list[float]:
        counts: dict[int, float] = {}
        for token in self._tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            counts[bucket] = counts.get(bucket, 0.0) + (1.0 if digest[4] & 1 else -1.0)

        vector = [0.0] * self.dimension
        for bucket, raw in counts.items():
            vector[bucket] = math.copysign(1.0 + math.log(abs(raw)), raw) if raw else 0.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


_provider: EmbeddingProvider | None = None


def get_embedding_provider(force: str | None = None) -> EmbeddingProvider:
    """Singleton - loading a transformer twice wastes hundreds of MB."""
    global _provider
    if _provider is not None and force is None:
        return _provider

    target = force or settings.embedding_model
    if target.lower() in {"hashing", "fallback", "none"}:
        _provider = HashingEmbeddingProvider()
        return _provider

    # The first load may download ~130 MB. A slow network is fine; a stalled
    # one would otherwise hang the app forever, so the load is bounded.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(SentenceTransformerProvider, target)
    try:
        _provider = future.result(timeout=settings.embedding_load_timeout)
        log.info("embeddings ready: %s (%d-d)", _provider.name, _provider.dimension)
    except concurrent.futures.TimeoutError:
        log.warning(
            "loading '%s' exceeded %ds - using the offline hashing embedder. "
            "Raise EMBEDDING_LOAD_TIMEOUT or pre-download the model.",
            target, settings.embedding_load_timeout,
        )
        _provider = HashingEmbeddingProvider()
    except Exception as exc:
        log.warning("could not load '%s' (%s) - using the offline hashing embedder",
                    target, exc)
        _provider = HashingEmbeddingProvider()
    finally:
        executor.shutdown(wait=False)
    return _provider


def reset_embedding_provider() -> None:
    global _provider
    _provider = None
