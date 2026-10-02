"""Document processing - one call from uploaded bytes to a searchable index.

WHAT   Upload -> Extract -> Clean -> Chunk -> Embed -> Store -> Register,
       with a per-stage progress callback the UI renders as a checklist.
WHY    The Documents page, the tests and the demo script all need exactly this
       sequence; having it in one function is what stops it drifting.
LIB    document_loader + chunker + embeddings + vector_store + registry.
IN     bytes/path
OUT    IngestOutcome(document, skipped, error)
NEXT   -> retriever / the whole agent stack

Dedup: the SHA-256 of the file bytes is the document_id.  An already-indexed
hash short-circuits before embedding, so re-uploading a 300-page PDF costs a
hash, not a GPU minute.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app.document_processing.document_loader import LoadedDocument, load_document, load_upload
from app.document_processing.metadata import get_registry
from app.document_processing.pdf_extractor import PDFExtractionError
from app.models.schemas import Document
from app.rag.chunker import chunk_pages
from app.rag.embeddings import get_embedding_provider
from app.rag.vector_store import get_vector_store, record_fingerprint
from app.utils.file_utils import UnsafeFileError, sha256_bytes, sha256_file
from app.utils.logging import get_logger

log = get_logger("pipeline")

STAGES = ["Uploading", "Extracting", "Chunking", "Embedding", "Indexing", "Ready"]
Progress = Optional[Callable[[str, str], None]]     # (stage, status) -> None


@dataclass
class IngestOutcome:
    document: Document | None = None
    skipped: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.document is not None and not self.error


def _tick(progress: Progress, stage: str, status: str = "done") -> None:
    if progress:
        try:
            progress(stage, status)
        except Exception:      # a UI callback must never break ingestion
            pass


def index_loaded(loaded: LoadedDocument, progress: Progress = None,
                 reindex: bool = False) -> IngestOutcome:
    registry = get_registry()

    if registry.has(loaded.document_id) and not reindex:
        existing = registry.get(loaded.document_id)
        log.info("skip: %s is already indexed (identical sha256)", loaded.name)
        _tick(progress, "Ready", "skipped")
        return IngestOutcome(document=existing, skipped=True)

    _tick(progress, "Extracting")

    # --- chunk ---
    _tick(progress, "Chunking", "running")
    chunks = chunk_pages(loaded.pages, loaded.document_id)
    if not chunks:
        return IngestOutcome(
            error="'" + loaded.name + "' produced no indexable text. "
                  "It may be a scanned PDF - enable OCR and retry."
        )
    _tick(progress, "Chunking")

    # --- embed ---
    _tick(progress, "Embedding", "running")
    provider = get_embedding_provider()
    try:
        vectors = provider.embed_documents([c.content for c in chunks])
    except Exception as exc:
        log.error("embedding failed for %s: %s", loaded.name, exc)
        return IngestOutcome(error="Embedding failed: " + str(exc))
    log.info("Embeddings generated (%d vectors, %s)", len(vectors), provider.name)
    _tick(progress, "Embedding")

    # --- store ---
    _tick(progress, "Indexing", "running")
    store = get_vector_store()
    if reindex:
        store.delete_document(loaded.document_id)
    try:
        store.add(chunks, vectors)
    except Exception as exc:
        log.error("vector store rejected %s: %s", loaded.name, exc)
        return IngestOutcome(error="Vector database error: " + str(exc))
    record_fingerprint(provider)
    log.info("Vector database updated (%d total vectors)", store.count())
    _tick(progress, "Indexing")

    # --- register ---
    document = Document(
        document_id=loaded.document_id,
        name=loaded.name,
        path=str(loaded.path),
        file_type=loaded.file_type,
        size_bytes=loaded.size_bytes,
        page_count=loaded.page_count,
        chunk_count=len(chunks),
        image_count=len(loaded.images),
        table_count=len(loaded.tables),
        title=loaded.metadata.get("title", ""),
        author=loaded.metadata.get("author", ""),
        ocr_pages=int(loaded.metadata.get("ocr_pages", 0) or 0),
    )
    registry.add(document)
    registry.add_images(loaded.document_id, loaded.images)
    _tick(progress, "Ready")
    return IngestOutcome(document=document)


def index_upload(data: bytes, filename: str, progress: Progress = None,
                 reindex: bool = False) -> IngestOutcome:
    """Full path for a Streamlit upload. Never raises - returns the error."""
    _tick(progress, "Uploading", "running")
    try:
        registry = get_registry()
        digest = sha256_bytes(data)
        if registry.has(digest) and not reindex:
            _tick(progress, "Uploading")
            _tick(progress, "Ready", "skipped")
            log.info("skip: %s already indexed", filename)
            return IngestOutcome(document=registry.get(digest), skipped=True)

        loaded = load_upload(data, filename)
        _tick(progress, "Uploading")
        return index_loaded(loaded, progress=progress, reindex=reindex)
    except (PDFExtractionError, UnsafeFileError) as exc:
        log.error("ingestion refused %s: %s", filename, exc)
        return IngestOutcome(error=str(exc))
    except Exception as exc:
        log.exception("unexpected ingestion failure for %s", filename)
        return IngestOutcome(error="Unexpected error while processing '" + filename + "': " + str(exc))


def index_path(path: str | Path, progress: Progress = None,
               reindex: bool = False) -> IngestOutcome:
    """Index a file already on disk (demo data, CLI, tests)."""
    path = Path(path)
    try:
        loaded = load_document(path, document_id=sha256_file(path))
        return index_loaded(loaded, progress=progress, reindex=reindex)
    except (PDFExtractionError, UnsafeFileError) as exc:
        return IngestOutcome(error=str(exc))
    except Exception as exc:
        log.exception("unexpected ingestion failure for %s", path)
        return IngestOutcome(error=str(exc))


def delete_document(document_id: str) -> int:
    """Remove a document from both the index and the registry."""
    removed = get_vector_store().delete_document(document_id)
    get_registry().remove(document_id)
    log.info("deleted document %s... (%d vectors dropped)", document_id[:10], removed)
    return removed
