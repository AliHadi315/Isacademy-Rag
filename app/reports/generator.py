"""Report generation - Markdown, HTML and PDF from an answered question.

A report is always built from a completed `AnswerResult`, so it can only
contain what the system actually retrieved and showed. Sections follow the
answer: question, answer, key findings, sources, visual analysis (only when
the question involved a visual), the semantic map and the cluster explanation.
"""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

from app.config import settings
from app.i18n import t
from app.models.schemas import AnswerResult
from app.utils.helpers import timestamp_slug
from app.utils.logging import get_logger

log = get_logger("reports")

TEMPLATES = Path(__file__).parent / "templates"
_FALLBACK_CSS = "body{font-family:system-ui,sans-serif;max-width:860px;margin:0 auto;padding:40px}"


def _slug(title: str) -> str:
    """Filename-safe, but keeps Arabic letters so Arabic reports keep their name."""
    cleaned = re.sub(r"[^\w-]+", "_", title, flags=re.UNICODE)
    return cleaned[:50].strip("_") or "report"


def _stylesheet() -> str:
    try:
        return (TEMPLATES / "report.css").read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("report.css unreadable (%s) - minimal styling", exc)
        return _FALLBACK_CSS


# ---------------------------------------------------------------------------
def build_sections(result: AnswerResult, lang: str | None = None) -> list:
    """[(title, body, [figure paths])] - the single source for all formats."""
    lang = lang or result.language
    sections: list = []

    sections.append((t("question", lang), result.query, []))
    sections.append((t("answer", lang), result.answer or "-", []))

    if result.key_findings:
        sections.append((
            t("key_findings", lang),
            "\n".join("- " + f for f in result.key_findings),
            [],
        ))

    sources = result.sources
    if sources:
        sections.append((
            t("sources", lang),
            "\n".join(str(i) + ". " + s for i, s in enumerate(sources, 1)),
            [],
        ))

    visual = result.visual
    if visual and visual.found:
        body = visual.analysis or visual.message or "-"
        body += "\n\n" + t("source", lang) + ": " + visual.citation
        figures = [visual.image_path] if Path(visual.image_path).exists() else []
        sections.append((t("visual_analysis", lang), body, figures))

    pca = result.pca
    if pca and pca.available:
        lines = [
            t("query_cluster", lang) + ": " + str(pca.query_cluster),
            t("closest_cluster", lang) + ": " + str(pca.closest_cluster),
            "",
        ]
        lines.append(t("closest_to_question", lang) + ":")
        for i, point in enumerate(pca.closest, 1):
            lines.append(
                str(i) + ". " + point.document + " — " + t("page", lang) + " "
                + str(point.page) + " — " + ("%.2f" % point.similarity)
            )
        if pca.nearby_other_cluster:
            lines += ["", t("other_nearby", lang) + ":"]
            lines += [
                "- " + p.document + " — " + t("page", lang) + " " + str(p.page)
                + " (" + t("cluster", lang) + " " + str(p.cluster) + ")"
                for p in pca.nearby_other_cluster
            ]
        sections.append((t("semantic_pca_map", lang), "\n".join(lines), []))

        if pca.explanation:
            sections.append((t("why_close", lang), pca.explanation, []))
        sections.append(("", t("pca_limitation", lang), []))

    return sections


# ---------------------------------------------------------------------------
def to_markdown(result: AnswerResult, lang: str | None = None) -> str:
    lang = lang or result.language
    lines = [
        "# " + t("reports_title", lang) + ": " + (result.query or "-"),
        "",
        "*" + result.created_at + "*",
        "",
        "---",
        "",
    ]
    for title, body, figures in build_sections(result, lang):
        if title:
            lines += ["## " + title, ""]
        lines += [body, ""]
        for figure in figures:
            name = Path(figure).name
            lines += ["![" + name + "](" + Path(figure).as_posix() + ")", ""]
    return "\n".join(lines).rstrip() + "\n"


def to_html(result: AnswerResult, lang: str | None = None) -> str:
    lang = lang or result.language
    rtl = lang == "ar"

    def esc(text) -> str:
        return html.escape(str(text)).replace("\n", "<br>")

    parts = [
        "<!doctype html><html lang='" + lang + "'"
        + (" dir='rtl'" if rtl else "") + "><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>" + html.escape(result.query or "Report") + "</title>",
        "<style>", _stylesheet(),
        ("body,.page{direction:rtl;text-align:right}" if rtl else ""),
        "</style></head><body><div class='page'>",
        "<h1>" + html.escape(result.query or "Report") + "</h1>",
        "<div class='meta'>" + html.escape(result.created_at) + "</div>",
    ]
    for title, body, figures in build_sections(result, lang):
        if title:
            parts.append("<h2>" + html.escape(title) + "</h2>")
        parts.append("<p>" + esc(body) + "</p>")
        for figure in figures:
            src = _img_src(figure)
            if src:
                parts.append(
                    "<figure><img src='" + src + "' alt='figure'>"
                    "<figcaption>" + html.escape(Path(figure).name)
                    + "</figcaption></figure>"
                )
    parts += ["</div></body></html>"]
    return "".join(parts)


def _img_src(path: str) -> str:
    file = Path(path)
    if not file.exists():
        log.warning("figure missing, skipped: %s", path)
        return ""
    try:
        mime = "image/png" if file.suffix.lower() == ".png" else "image/jpeg"
        return "data:" + mime + ";base64," + base64.b64encode(file.read_bytes()).decode("ascii")
    except OSError as exc:
        log.warning("could not embed %s: %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
def render(result: AnswerResult, formats: list, lang: str | None = None) -> dict:
    """Write the requested formats. One failing format never blocks the others."""
    lang = lang or result.language
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    base = "report_" + _slug(result.query or "question") + "_" + timestamp_slug()
    written: dict = {}
    wanted = {str(f).lower() for f in formats}

    if wanted & {"md", "markdown"}:
        try:
            path = settings.output_dir / (base + ".md")
            path.write_text(to_markdown(result, lang), encoding="utf-8")
            written["markdown"] = str(path)
        except OSError as exc:
            log.error("markdown report failed: %s", exc)

    if "html" in wanted:
        try:
            path = settings.output_dir / (base + ".html")
            path.write_text(to_html(result, lang), encoding="utf-8")
            written["html"] = str(path)
        except OSError as exc:
            log.error("HTML report failed: %s", exc)

    if "pdf" in wanted:
        try:
            from app.reports.pdf_generator import to_pdf

            written["pdf"] = to_pdf(
                result.query or "Report",
                build_sections(result, lang),
                settings.output_dir / (base + ".pdf"),
                created_at=result.created_at,
            )
        except ImportError:
            written["pdf_error"] = "PDF export needs reportlab (pip install reportlab)."
        except Exception as exc:
            log.exception("PDF report failed")
            written["pdf_error"] = str(exc)

    for kind, path in written.items():
        if "error" not in kind:
            log.info("Report written: %s -> %s", kind, Path(path).name)
    return written


def list_reports() -> list:
    files = [
        p for p in settings.output_dir.glob("report_*")
        if p.suffix.lower() in {".md", ".html", ".pdf"}
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
