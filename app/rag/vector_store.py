"""ChromaDB vector store.

Stores chunk text + embedding + metadata (document, page, content_type,
chunk_id) and persists to CHROMA_PATH.

Two kinds of read:
  * `query()`   - top-K nearest chunks, for retrieval.
  * `neighborhood()` - the same search but returning the raw embedding vectors
    too, which is what the semantic PCA projects. Fetching vectors here (rather
    than re-embedding text later) guarantees the query and the documents live
    in exactly the same space.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.models.schemas import Chunk, SearchResult
from app.utils.logging import get_logger

log = get_logger("vectorstore")

COLLECTION = "isacademy_documents"


class VectorStoreError(RuntimeError):
    pass


def _to_result(text: str, meta: dict, distance: float) -> SearchResult:
    meta = meta or {}
    return SearchResult(
        content=text or "",
        document=str(meta.get("document_name", "unknown")),
        document_id=str(meta.get("document_id", "")),
        page=int(meta.get("page_number", 0) or 0),
        chunk_id=str(meta.get("chunk_id", "")),
        content_type=str(meta.get("content_type", "text")),
        image_path=str(meta.get("image_path", "") or ""),
        # Chroma reports cosine *distance*; the app works in similarity.
        score=round(max(0.0, 1.0 - float(distance)), 4),
    )


class VectorStore:
    """Thin wrapper over a persistent Chroma collection."""

    backend = "chromadb"

    def __init__(self, path: Path | None = None):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.path = Path(path or settings.chroma_path)
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            raise VectorStoreError("Could not open the vector database: " + str(exc)) from exc
        log.info("ChromaDB ready at %s (%d vectors)", self.path, self.count())

    # ------------------------------------------------------------------
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )
        return len(chunks)

    def query(self, embedding: list[float], top_k: int,
              document_ids: list[str] | None = None) -> list[SearchResult]:
        raw = self._raw_query(embedding, top_k, document_ids, with_vectors=False)
        return [
            _to_result(doc, meta, dist)
            for doc, meta, dist in zip(raw["documents"], raw["metadatas"], raw["distances"])
        ]

    def neighborhood(self, embedding: list[float], size: int,
                     document_ids: list[str] | None = None):
        """(results, vectors) - the nearest `size` chunks WITH their vectors."""
        raw = self._raw_query(embedding, size, document_ids, with_vectors=True)
        results = [
            _to_result(doc, meta, dist)
            for doc, meta, dist in zip(raw["documents"], raw["metadatas"], raw["distances"])
        ]
        return results, [list(v) for v in raw["embeddings"]]

    # ------------------------------------------------------------------
    def _raw_query(self, embedding, n: int, document_ids, with_vectors: bool) -> dict:
        empty = {"documents": [], "metadatas": [], "distances": [], "embeddings": []}
        total = self.count()
        if total == 0:
            return empty

        where = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": list(document_ids)}}
            )
        include = ["documents", "metadatas", "distances"]
        if with_vectors:
            include.append("embeddings")

        try:
            raw = self.collection.query(
                query_embeddings=[embedding],
                n_results=max(1, min(n, total)),
                where=where,
                include=include,
            )
        except Exception as exc:
            raise VectorStoreError("Vector search failed: " + str(exc)) from exc

        out = {
            "documents": raw["documents"][0],
            "metadatas": raw["metadatas"][0],
            "distances": raw["distances"][0],
            "embeddings": [],
        }
        if with_vectors and raw.get("embeddings") is not None:
            out["embeddings"] = raw["embeddings"][0]
        return out

    # ------------------------------------------------------------------
    def delete_document(self, document_id: str) -> int:
        before = self.count()
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception as exc:
            log.error("delete failed for %s: %s", document_id[:10], exc)
            return 0
        return before - self.count()

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def reset(self) -> None:
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )


# ---------------------------------------------------------------------------
# Embedding fingerprint
#
# Vectors written by one embedding model are meaningless to another. If the
# model changes after indexing, queries silently return nonsense - so the
# mismatch is made explicit, with the fix.
# ---------------------------------------------------------------------------
def _fingerprint_file() -> Path:
    return Path(settings.chroma_path) / "embedding_fingerprint.json"


def record_fingerprint(provider) -> None:
    try:
        _fingerprint_file().write_text(
            json.dumps({"provider": provider.name, "dimension": provider.dimension}),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("could not write the embedding fingerprint: %s", exc)


def check_fingerprint(provider) -> str:
    """Return "" when compatible, otherwise a message explaining the mismatch."""
    path = _fingerprint_file()
    if not path.exists() or get_vector_store().count() == 0:
        return ""
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if stored.get("provider") == provider.name:
        return ""
    return (
        "The index was built with the embedding model '" + str(stored.get("provider"))
        + "' but '" + provider.name + "' is loaded now. Vectors from different models "
        "are not comparable - re-index your documents."
    )


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_vector_store() -> None:
    global _store
    _store = None
