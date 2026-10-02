"""PDF extraction (PyMuPDF).

WHAT   Turns a PDF file into a list of `DocumentPage` objects carrying text,
       embedded images and detected tables, with the page number preserved on
       every artefact so citations can never be invented later.
WHY    Page provenance is the anchor of the whole anti-hallucination story: a
       chunk that does not know its page cannot be cited honestly.
LIB    PyMuPDF (fitz) - fast, pure-wheel, extracts text + images + table finder
       in one pass.  Optional easyocr fallback for scanned pages.
IN     Path to a .pdf
OUT    (list[DocumentPage], metadata dict)
NEXT   -> document_loader -> chunker -> embeddings -> vector_store
"""
from __future__ import annotations

import re
from pathlib import Path

from app.config import settings
from app.models.schemas import DocumentPage
from app.utils.helpers import clean_text
from app.utils.logging import get_logger

from .image_extractor import extract_page_images
from .table_extractor import extract_page_tables

log = get_logger("pdf")

MIN_TEXT_CHARS_BEFORE_OCR = 40


class PDFExtractionError(RuntimeError):
    """Raised for unreadable / corrupt / encrypted / empty PDFs."""


def _fitz():
    """PyMuPDF, imported under whichever name this version exposes."""
    try:
        import pymupdf                     # modern name (>= 1.24.3)

        return pymupdf
    except ImportError:
        pass
    try:
        import fitz                        # legacy name

        return fitz
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PDFExtractionError("PyMuPDF is not installed. Run: pip install PyMuPDF") from exc


def extract_pdf(path: str | Path, document_name: str | None = None, with_images: bool = True,
                with_tables: bool = True) -> tuple[list[DocumentPage], dict]:
    """Extract every page of a PDF. Raises PDFExtractionError on bad input."""
    fitz = _fitz()
    path = Path(path)
    name = document_name or path.name

    if not path.exists():
        raise PDFExtractionError("File not found: " + str(path))
    if path.stat().st_size == 0:
        raise PDFExtractionError("File is empty: " + name)

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise PDFExtractionError("Could not open '" + name + "' - it may be corrupted.") from exc

    if doc.needs_pass:
        doc.close()
        raise PDFExtractionError("'" + name + "' is password-protected.")
    if doc.page_count == 0:
        doc.close()
        raise PDFExtractionError("'" + name + "' contains no pages.")

    meta = dict(doc.metadata or {})
    metadata = {
        "title": (meta.get("title") or "").strip() or path.stem,
        "author": (meta.get("author") or "").strip(),
        "page_count": doc.page_count,
        "producer": (meta.get("producer") or "").strip(),
    }

    log.info("PDF extraction started: %s (%d pages)", name, doc.page_count)
    pages: list[DocumentPage] = []
    ocr_pages = 0

    for index in range(doc.page_count):
        page_no = index + 1
        try:
            page = doc.load_page(index)
            text = clean_text(page.get_text("text"))
            ocr_used = False

            if len(text) < MIN_TEXT_CHARS_BEFORE_OCR and settings.enable_ocr:
                ocr_text = _ocr_page(page, page_no, name)
                if ocr_text:
                    text, ocr_used = ocr_text, True
                    ocr_pages += 1

            images = extract_page_images(page, name, page_no) if with_images else []
            tables = extract_page_tables(page, name, page_no) if with_tables else []

            pages.append(
                DocumentPage(
                    document_name=name,
                    page_number=page_no,
                    text=text,
                    images=images,
                    tables=tables,
                    ocr_used=ocr_used,
                )
            )
        except Exception as exc:  # one bad page must not kill the document
            log.warning("Page %d of %s failed: %s", page_no, name, exc)
            pages.append(DocumentPage(document_name=name, page_number=page_no, text=""))

    doc.close()
    metadata["ocr_pages"] = ocr_pages
    total_chars = sum(len(p.text) for p in pages)
    log.info(
        "%d pages extracted from %s (%d chars, %d images, %d tables)",
        len(pages), name, total_chars,
        sum(len(p.images) for p in pages), sum(len(p.tables) for p in pages),
    )

    if total_chars == 0 and not any(p.images for p in pages):
        raise PDFExtractionError(
            "'" + name + "' produced no text or images. If it is a scanned PDF, "
            "set ENABLE_OCR=true and install easyocr."
        )
    return pages, metadata


def _ocr_page(page, page_no: int, name: str) -> str:
    """Optional OCR fallback for scanned pages. Never fatal."""
    try:
        import numpy as np
        import easyocr
    except ImportError:
        log.warning("ENABLE_OCR=true but easyocr is not installed - skipping OCR.")
        return ""
    try:
        global _READER
        try:
            _READER
        except NameError:
            _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        text = " ".join(_READER.readtext(img, detail=0))
        log.info("OCR recovered %d chars on page %d of %s", len(text), page_no, name)
        return clean_text(text)
    except Exception as exc:
        log.warning("OCR failed on page %d of %s: %s", page_no, name, exc)
        return ""


def render_page_image(pdf_path, page_number: int, dpi: int = 150):
    """Render one whole PDF page to a PNG and return its path.

    This is the fallback that makes the "always show the visual" rule keepable:
    when a chart cannot be isolated as an embedded image (it was drawn with
    vector graphics, or spans the page), showing the page itself is far more
    useful than a text-only answer.
    """
    from app.config import settings

    fitz = _fitz()
    pdf_path = Path(pdf_path)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", pdf_path.stem)[:60]
    out_path = settings.images_dir / (safe + "_page" + str(page_number) + "_render.png")
    if out_path.exists():
        return out_path

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise PDFExtractionError("Could not open " + pdf_path.name + ": " + str(exc)) from exc
    try:
        if not 1 <= page_number <= doc.page_count:
            raise PDFExtractionError(
                "Page " + str(page_number) + " does not exist in " + pdf_path.name
                + " (it has " + str(doc.page_count) + " pages)."
            )
        page = doc.load_page(page_number - 1)
        page.get_pixmap(dpi=dpi).save(str(out_path))
    finally:
        doc.close()
    log.info("rendered %s page %d -> %s", pdf_path.name, page_number, out_path.name)
    return out_path


def extract_text_file(path: str | Path, document_name: str | None = None):
    """.txt / .md are treated as a single-page document."""
    path = Path(path)
    name = document_name or path.name
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        raise PDFExtractionError("'" + name + "' is empty.")
    pages = [DocumentPage(document_name=name, page_number=1, text=clean_text(raw))]
    return pages, {"title": path.stem, "author": "", "page_count": 1, "ocr_pages": 0}
