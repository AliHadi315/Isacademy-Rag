"""Professional PDF generation.

Renders `[(section title, body, [figure paths])]` to a paginated A4 PDF with a
title page, numbered sections, embedded figures scaled to the text width, and a
page-numbered footer.

Arabic is supported: `app/reports/arabic.py` registers a system font that has
the glyphs, joins the letters and applies right-to-left ordering, and the
paragraph styles switch to that font whenever Arabic is detected.
"""
from __future__ import annotations

import html
from pathlib import Path

from app.reports.arabic import has_arabic, register_arabic_font, shape
from app.utils.logging import get_logger

log = get_logger("pdf")

INK = "#0f172a"
MUTED = "#475569"
ACCENT = "#2563eb"
LINE = "#cbd5e1"


def _styles():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "IsaTitle", parent=base["Title"], fontSize=25, leading=30,
            textColor=colors.HexColor(INK), spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "IsaSubtitle", parent=base["Normal"], fontSize=11, leading=16,
            textColor=colors.HexColor(MUTED), alignment=1, spaceAfter=26,
        ),
        "h2": ParagraphStyle(
            "IsaH2", parent=base["Heading2"], fontSize=14.5, leading=19,
            textColor=colors.HexColor(ACCENT), spaceBefore=18, spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "IsaBody", parent=base["BodyText"], fontSize=10, leading=15.5,
            textColor=colors.HexColor(INK), alignment=TA_JUSTIFY, spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "IsaBullet", parent=base["BodyText"], fontSize=9.6, leading=14,
            leftIndent=14, bulletIndent=4, spaceAfter=4,
            textColor=colors.HexColor(INK),
        ),
        "flag": ParagraphStyle(
            "IsaFlag", parent=base["BodyText"], fontSize=9.6, leading=14,
            leftIndent=14, bulletIndent=4, spaceAfter=4,
            textColor=colors.HexColor("#b91c1c"),
        ),
        "caption": ParagraphStyle(
            "IsaCaption", parent=base["Normal"], fontSize=8.4, leading=11,
            alignment=1, textColor=colors.HexColor(MUTED), spaceAfter=14,
        ),
        "toc": ParagraphStyle(
            "IsaToc", parent=base["Normal"], fontSize=10, leading=16,
            leftIndent=12, textColor=colors.HexColor(INK),
        ),
    }


def _clean(text) -> str:
    """Shape Arabic, escape for reportlab's mini-HTML, keep line breaks."""
    return html.escape(shape(str(text))).replace("\n", "<br/>")


def _arabic_styles(styles: dict) -> dict:
    """Swap in an Arabic-capable font and right-align the text styles."""
    from reportlab.lib.enums import TA_RIGHT

    font = register_arabic_font()
    if not font:
        return styles
    for key in ("title", "subtitle", "h2", "body", "bullet", "flag", "caption"):
        style = styles[key]
        style.fontName = font
        if key in ("h2", "body", "bullet", "flag"):
            style.alignment = TA_RIGHT
    return styles


def _footer(canvas, doc, title: str):
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    canvas.saveState()
    width, _ = canvas._pagesize
    canvas.setStrokeColor(colors.HexColor(LINE))
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.5 * cm, width - 2 * cm, 1.5 * cm)

    # Helvetica has no Arabic glyphs; an Arabic title would print as boxes.
    footer_font = "Helvetica"
    if has_arabic(title):
        footer_font = register_arabic_font() or "Helvetica"
        title = shape(title)
    canvas.setFont(footer_font, 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(2 * cm, 1.05 * cm, title[:80])
    canvas.drawRightString(width - 2 * cm, 1.05 * cm, "Page " + str(canvas.getPageNumber()))
    canvas.restoreState()


def to_pdf(title: str, sections: list, out_path, created_at: str = "") -> str:
    """Render `[(section title, body, [figure paths])]` to a paginated A4 PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        HRFlowable,
        Image,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
    )
    from reportlab.lib.utils import ImageReader

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    # If any section contains Arabic, render the whole document with a font
    # that has the glyphs - mixing fonts mid-report looks worse than one.
    if has_arabic(title) or any(
        has_arabic(t) or has_arabic(b) for t, b, _ in sections
    ):
        styles = _arabic_styles(styles)
        log.info("Arabic detected - using an Arabic-capable font for this PDF")

    doc = BaseDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2.2 * cm,
        title=title, author="Isacademy AI",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([
        PageTemplate(id="main", frames=[frame],
                     onPage=lambda c, d: _footer(c, d, title))
    ])

    story = [
        Spacer(1, 3.4 * cm),
        Paragraph(_clean(title), styles["title"]),
        HRFlowable(width="60%", thickness=1.4, color=colors.HexColor(ACCENT),
                   spaceBefore=6, spaceAfter=14, hAlign="CENTER"),
        Paragraph("Isacademy AI - Document Intelligence Report", styles["subtitle"]),
        Paragraph("Generated " + _clean(created_at), styles["subtitle"]),
        Spacer(1, 1.6 * cm),
        Paragraph(
            "Every source cited in this report was actually retrieved from the "
            "indexed documents.",
            styles["caption"],
        ),
        PageBreak(),
    ]

    number = 0
    for section_title, body, figures in sections:
        if section_title:
            number += 1
            story.append(Paragraph(_clean(str(number) + ". " + section_title), styles["h2"]))

        for line in str(body or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ", "• ")):
                story.append(
                    Paragraph(_clean(line[2:].strip()), styles["bullet"], bulletText="•")
                )
            else:
                story.append(Paragraph(_clean(line), styles["body"]))

        for figure in figures:
            path = Path(figure)
            if not path.exists():
                log.warning("figure missing, skipped in PDF: %s", figure)
                continue
            try:
                width_px, height_px = ImageReader(str(path)).getSize()
                width = min(doc.width, 15.5 * cm)
                height = width * height_px / max(width_px, 1)
                if height > 17 * cm:                       # keep it on one page
                    height, width = 17 * cm, 17 * cm * width_px / max(height_px, 1)
                story += [
                    Spacer(1, 0.3 * cm),
                    Image(str(path), width=width, height=height),
                    Paragraph(_clean(path.name), styles["caption"]),
                ]
            except Exception as exc:
                log.warning("could not embed %s in the PDF: %s", path.name, exc)

    doc.build(story)
    log.info("PDF report written: %s", out_path.name)
    return str(out_path)
