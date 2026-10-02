"""Upload intake: validate -> store safely -> hash -> extract.

WHAT   The single door every file enters through.  Validates type and size,
       writes it under a sanitised name inside data/uploads, hashes the bytes
       for dedup, and dispatches to the right extractor.
WHY    Centralising this is what makes the security rules (spec 45) and the
       "never re-embed the same file" rule (spec 46) enforceable in one place.
LIB    stdlib + the pdf extractor.
IN     raw bytes + original filename  (or an existing path)
OUT    LoadedDocument(pages, metadata, document_id, path)
NEXT   -> rag.pipeline.index_document
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.models.schemas import DocumentPage
from app.utils.file_utils import (
    UnsafeFileError,
    resolve_inside,
    safe_filename,
    sha256_bytes,
    sha256_file,
    validate_upload,
)
from app.utils.logging import get_logger

from .pdf_extractor import PDFExtractionError, extract_pdf, extract_text_file

log = get_logger("loader")


@dataclass
class LoadedDocument:
    document_id: str
    name: str
    path: Path
    file_type: str
    size_bytes: int
    pages: list[DocumentPage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def images(self) -> list:
        return [img for p in self.pages for img in p.images]

    @property
    def tables(self) -> list:
        return [t for p in self.pages for t in p.tables]


def save_upload(data: bytes, filename: str) -> tuple[Path, str]:
    """Validate and persist an uploaded file. Returns (path, sha256)."""
    validate_upload(filename, len(data), settings.allowed_doc_ext, settings.max_upload_mb)
    target = resolve_inside(settings.upload_dir, filename)
    target.write_bytes(data)
    digest = sha256_bytes(data)
    log.info("User uploaded %s (%d bytes, sha256=%s...)", target.name, len(data), digest[:10])
    return target, digest


def save_dataset(data: bytes, filename: str) -> Path:
    """Same rules, different allow-list, for CSV/Excel datasets."""
    validate_upload(filename, len(data), settings.allowed_data_ext, settings.max_upload_mb)
    target = resolve_inside(settings.datasets_dir, filename)
    target.write_bytes(data)
    log.info("Dataset uploaded: %s (%d bytes)", target.name, len(data))
    return target


def load_document(path: str | Path, document_id: str | None = None,
                  document_name: str | None = None) -> LoadedDocument:
    """Extract an already-saved file into pages. Raises PDFExtractionError."""
    path = Path(path)
    if not path.exists():
        raise PDFExtractionError("File not found: " + str(path))

    name = document_name or path.name
    suffix = path.suffix.lower()
    if suffix not in settings.allowed_doc_ext:
        raise UnsafeFileError("Unsupported document type: " + (suffix or "none"))

    digest = document_id or sha256_file(path)
    if suffix == ".pdf":
        pages, meta = extract_pdf(path, document_name=name)
    else:
        pages, meta = extract_text_file(path, document_name=name)

    return LoadedDocument(
        document_id=digest,
        name=safe_filename(name),
        path=path,
        file_type=suffix.lstrip("."),
        size_bytes=path.stat().st_size,
        pages=pages,
        metadata=meta,
    )


def load_upload(data: bytes, filename: str) -> LoadedDocument:
    """save_upload + load_document, the path the Streamlit uploader takes."""
    path, digest = save_upload(data, filename)
    return load_document(path, document_id=digest, document_name=path.name)
