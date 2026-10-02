"""Build the Code Guide PDF.

    python docs/build_code_guide.py

Reads the structured content in `docs/code_guide_content.py` and renders it to
`docs/Isacademy_Code_Guide.pdf` - a cover page, an automatic table of
contents, numbered sections, code blocks, tables and callouts.

The content is a plain list of (kind, payload) tuples, so updating the guide
means editing text, never layout code.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from app.reports.arabic import has_arabic, register_arabic_font, shape  # noqa: E402

from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

INK = "#0f172a"
MUTED = "#475569"
ACCENT = "#1d4ed8"
ACCENT_SOFT = "#eff6ff"
LINE = "#cbd5e1"
CODE_BG = "#f8fafc"
NOTE_BG = "#fef9c3"

OUT = ROOT / "docs" / "Isacademy_Code_Guide.pdf"


# ---------------------------------------------------------------------------
ARABIC_FONT = register_arabic_font()


def ar(text: str) -> str:
    """Shape Arabic runs so they render joined and right-to-left.

    Mixed English/Arabic sentences are handled by shaping only the Arabic
    stretches, so the surrounding English keeps its normal direction.
    """
    if not has_arabic(text):
        return text
    import re as _re

    return _re.sub(
        r"[؀-ۿݐ-ݿ][؀-ۿݐ-ݿ\s،؟.,:\-]*",
        lambda m: shape(m.group(0)),
        text,
    )


def build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontSize=30, leading=36,
            textColor=colors.HexColor(INK), spaceAfter=6,
        ),
        "cover_sub": ParagraphStyle(
            "CoverSub", parent=base["Normal"], fontSize=13, leading=19,
            alignment=TA_CENTER, textColor=colors.HexColor(MUTED), spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontSize=19, leading=24,
            textColor=colors.HexColor(ACCENT), spaceBefore=6, spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=13.5, leading=18,
            textColor=colors.HexColor(INK), spaceBefore=14, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontSize=11.2, leading=15,
            textColor=colors.HexColor(ACCENT), spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontSize=9.8, leading=14.6,
            alignment=TA_JUSTIFY, textColor=colors.HexColor(INK), spaceAfter=7,
            fontName=ARABIC_FONT or "Helvetica",
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontSize=9.6, leading=14,
            leftIndent=16, bulletIndent=5, spaceAfter=3.5,
            textColor=colors.HexColor(INK),
            fontName=ARABIC_FONT or "Helvetica",
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"], fontName="Courier", fontSize=7.6,
            leading=9.8, textColor=colors.HexColor("#0b2447"),
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontSize=8.2, leading=11,
            textColor=colors.HexColor(MUTED), spaceAfter=10,
            fontName=ARABIC_FONT or "Helvetica",
        ),
        "toc": ParagraphStyle(
            "Toc", parent=base["Normal"], fontSize=9.6, leading=16,
            textColor=colors.HexColor(INK),
        ),
        "toc_sub": ParagraphStyle(
            "TocSub", parent=base["Normal"], fontSize=9, leading=14,
            leftIndent=18, textColor=colors.HexColor(MUTED),
        ),
        "note": ParagraphStyle(
            "Note", parent=base["BodyText"], fontSize=9.3, leading=13.6,
            textColor=colors.HexColor("#713f12"), spaceAfter=4,
            fontName=ARABIC_FONT or "Helvetica",
        ),
    }


def footer(canvas, doc):
    canvas.saveState()
    width, _ = canvas._pagesize
    page = canvas.getPageNumber()
    if page > 1:
        canvas.setStrokeColor(colors.HexColor(LINE))
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, 1.45 * cm, width - 2 * cm, 1.45 * cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(2 * cm, 1.02 * cm,
                          "Isacademy AI - Code Guide")
        canvas.drawRightString(width - 2 * cm, 1.02 * cm, str(page))
    canvas.restoreState()


# ---------------------------------------------------------------------------
def code_block(text: str, styles) -> Table:
    """A monospaced block on a tinted background with a left accent rule."""
    body = Preformatted(text.strip("\n"), styles["code"])
    table = Table([[body]], colWidths=[16.4 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(CODE_BG)),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor(ACCENT)),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def note_block(text: str, styles) -> Table:
    table = Table([[Paragraph(text, styles["note"])]], colWidths=[16.4 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(NOTE_BG)),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#facc15")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def shape_cell(value) -> str:
    from docs.build_code_guide import ar as _ar  # self-reference is fine here

    return _ar(str(value))


def data_table(rows: list, styles, widths=None) -> Table:
    """First row is the header."""
    wrapped = [
        [Paragraph("<b>" + shape_cell(c) + "</b>", styles["caption"]) for c in rows[0]]
    ] + [
        [Paragraph(shape_cell(c), styles["caption"]) for c in row] for row in rows[1:]
    ]
    table = Table(wrapped, colWidths=widths or None, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT_SOFT)),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


# ---------------------------------------------------------------------------
def build(content: list, subtitle: str = "") -> str:
    styles = build_styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.3 * cm, rightMargin=2.3 * cm,
        topMargin=2.2 * cm, bottomMargin=2.1 * cm,
        title="Isacademy AI - Code Guide",
        author="Isacademy Capstone",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])

    story: list = []

    # ---- cover ----
    story += [
        Spacer(1, 4.2 * cm),
        Paragraph("Isacademy AI", styles["cover_title"]),
        Paragraph("Document Intelligence with Gemini", styles["cover_sub"]),
        HRFlowable(width="45%", thickness=1.6, color=colors.HexColor(ACCENT),
                   spaceBefore=10, spaceAfter=18, hAlign="CENTER"),
        Paragraph("<b>Complete Code Guide</b>", styles["cover_sub"]),
        Paragraph(
            "Every folder, every class, every step - what it is, what it does, "
            "and how the pieces fit together.", styles["cover_sub"],
        ),
        Spacer(1, 3.5 * cm),
        Paragraph(subtitle, styles["cover_sub"]),
        PageBreak(),
    ]

    # ---- table of contents (built from the h1/h2 entries) ----
    story.append(Paragraph("Contents", styles["h1"]))
    number = 0
    for kind, payload in content:
        if kind == "h1":
            number += 1
            story.append(Paragraph(
                "<b>" + str(number) + ".</b>&nbsp;&nbsp;" + payload, styles["toc"]))
        elif kind == "h2":
            story.append(Paragraph(payload, styles["toc_sub"]))
    story.append(PageBreak())

    # ---- body ----
    number = 0
    for kind, payload in content:
        if kind == "h1":
            number += 1
            story += [
                CondPageBreak(6 * cm),
                Paragraph(str(number) + ". " + payload, styles["h1"]),
                HRFlowable(width="100%", thickness=0.8,
                           color=colors.HexColor(LINE), spaceAfter=10),
            ]
        elif kind == "h2":
            story.append(KeepTogether([
                CondPageBreak(3.5 * cm),
                Paragraph(payload, styles["h2"]),
            ]))
        elif kind == "h3":
            story.append(Paragraph(payload, styles["h3"]))
        elif kind == "p":
            story.append(Paragraph(ar(payload), styles["body"]))
        elif kind == "bullets":
            for item in payload:
                story.append(Paragraph(ar(item), styles["bullet"], bulletText="•"))
            story.append(Spacer(1, 5))
        elif kind == "steps":
            for i, item in enumerate(payload, 1):
                story.append(Paragraph(ar(item), styles["bullet"],
                                       bulletText=str(i) + "."))
            story.append(Spacer(1, 5))
        elif kind == "code":
            story += [code_block(payload, styles), Spacer(1, 9)]
        elif kind == "note":
            story += [note_block(ar(payload), styles), Spacer(1, 9)]
        elif kind == "table":
            story += [data_table(payload, styles), Spacer(1, 10)]
        elif kind == "table_w":
            rows, widths = payload
            story += [data_table(rows, styles, widths), Spacer(1, 10)]
        elif kind == "caption":
            story.append(Paragraph(payload, styles["caption"]))
        elif kind == "pagebreak":
            story.append(PageBreak())
        elif kind == "space":
            story.append(Spacer(1, payload))
        else:
            raise ValueError("unknown content kind: " + kind)

    doc.build(story)
    return str(OUT)


def main() -> None:
    from datetime import date

    from docs.code_guide_content import CONTENT

    path = build(CONTENT, subtitle="Generated " + date.today().isoformat())
    size = Path(path).stat().st_size
    print("written:", path)
    print("size   :", round(size / 1024), "KB")


if __name__ == "__main__":
    main()
