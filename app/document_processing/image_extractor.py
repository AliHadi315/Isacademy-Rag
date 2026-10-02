"""Embedded-image extraction + caption harvesting.

WHAT   Pulls raster images out of a PDF page, writes them to data/images/ and
       tries to attach the nearby "Figure N: ..." caption.
WHY    The Vision agent needs real image files on disk plus a caption hint so
       a question like "what does Figure 3 show?" can find the right picture.
LIB    PyMuPDF (extraction) - Pillow only indirectly, via PyMuPDF's Pixmap.
IN     a fitz.Page, document name, page number
OUT    list[ExtractedImage]
NEXT   -> vision_agent
"""
from __future__ import annotations

import re

from app.config import settings
from app.models.schemas import ExtractedImage
from app.utils.logging import get_logger

log = get_logger("images")

MIN_DIM = 80          # ignore bullets, rules, logos
MAX_IMAGES_PER_PAGE = 12

CAPTION_RE = re.compile(
    r"^\s*(fig(?:ure)?|chart|table|diagram|exhibit|plot)\s*\.?\s*(\d+[a-z]?)\s*[.:\-–]?\s*(.{0,180})",
    re.I | re.M,
)


def page_captions(text: str) -> list[str]:
    """All 'Figure 3: ...'-style lines on a page, in order."""
    out = []
    for m in CAPTION_RE.finditer(text or ""):
        label = m.group(1).title() + " " + m.group(2)
        rest = " ".join(m.group(3).split())
        out.append((label + ": " + rest).strip(": ").strip())
    return out


def extract_page_images(page, document_name: str, page_number: int) -> list[ExtractedImage]:
    images: list[ExtractedImage] = []
    try:
        from .pdf_extractor import _fitz

        fitz = _fitz()
    except Exception:
        return images

    try:
        raw_text = page.get_text("text")
    except Exception:
        raw_text = ""
    captions = page_captions(raw_text)

    try:
        infos = page.get_images(full=True)
    except Exception as exc:
        log.warning("image listing failed p%d: %s", page_number, exc)
        return images

    doc = page.parent
    safe_doc = re.sub(r"[^A-Za-z0-9_-]+", "_", document_name)[:60]

    for i, info in enumerate(infos[:MAX_IMAGES_PER_PAGE]):
        xref = info[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.width < MIN_DIM or pix.height < MIN_DIM:
                pix = None
                continue
            if pix.n - pix.alpha >= 4:                 # CMYK -> RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_id = safe_doc + "_p" + str(page_number) + "_i" + str(i)
            out_path = settings.images_dir / (image_id + ".png")
            pix.save(str(out_path))
            images.append(
                ExtractedImage(
                    image_id=image_id,
                    document_name=document_name,
                    page_number=page_number,
                    path=str(out_path),
                    width=pix.width,
                    height=pix.height,
                    caption=captions[i] if i < len(captions) else (captions[0] if captions else ""),
                )
            )
            pix = None
        except Exception as exc:
            log.warning("image %d on p%d failed: %s", i, page_number, exc)
    return images
