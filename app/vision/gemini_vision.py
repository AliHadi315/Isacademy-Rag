"""Visual question answering with Gemini.

Two jobs:

1. **Find the visual.** An explicit "Figure 3" or "page 15" wins; otherwise the
   retrieved chunks decide. If no figure was extracted separately (charts drawn
   as vector graphics often are not), the whole page is rendered instead - a
   real page image always beats a text-only answer.

2. **Analyse it.** The image plus the user's question plus the surrounding
   document text go to Gemini, which is told to describe only what is visible.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.config import settings
from app.document_processing.metadata import get_registry
from app.document_processing.pdf_extractor import PDFExtractionError, render_page_image
from app.models.gemini import GeminiError, get_gemini
from app.models.schemas import ExtractedImage, QueryPlan, RetrievalResult, VisualAnswer
from app.utils.helpers import keywords, truncate
from app.utils.logging import get_logger

log = get_logger("vision")

VISION_PROMPT = {
    "en": """Analyze this visual in the context of the user's question.

User question:
{question}

Surrounding document text (context):
{context}

Explain only what can be supported by the visual and the provided document
context.

If the visual contains a chart, describe:
- axes and their units
- categories or series
- trends
- highest/lowest values when readable
- important relationships

If the visual contains a table:
- identify important values
- summarize patterns
- mention relevant rows/columns

Do not invent values that cannot be read. If something is unreadable, say so.
Answer in at most 10 short lines.""",
    "ar": """حلّل هذا العنصر المرئي في سياق سؤال المستخدم.

سؤال المستخدم:
{question}

النص المحيط من المستند (السياق):
{context}

اشرح فقط ما يمكن دعمه بالعنصر المرئي وسياق المستند المقدَّم.

إذا كان العنصر رسماً بيانياً، صِف:
- المحاور ووحداتها
- الفئات أو السلاسل
- الاتجاهات
- أعلى وأدنى القيم إذا كانت مقروءة
- العلاقات المهمة

إذا كان جدولاً:
- حدّد القيم المهمة
- لخّص الأنماط
- اذكر الصفوف والأعمدة ذات الصلة

لا تخترع قيماً لا يمكن قراءتها. إذا كان شيء غير مقروء، قل ذلك.
أجب في عشرة أسطر قصيرة كحد أقصى.""",
}


# ---------------------------------------------------------------------------
def _document_path(document_name: str) -> Path | None:
    doc = get_registry().by_name(document_name)
    if not doc:
        return None
    path = Path(doc.path)
    return path if path.exists() and path.suffix.lower() == ".pdf" else None


def _score_image(image: ExtractedImage, figure_hint: str, terms: set,
                 hot_pages: set) -> float:
    score = 0.0
    caption = (image.caption or "").lower()
    if figure_hint and re.search(r"\b" + re.escape(figure_hint.lower()) + r"\b", caption):
        score += 5.0
    if terms and caption:
        score += 2.0 * len(terms & set(keywords(caption, 20))) / max(len(terms), 1)
    if (image.document_name, image.page_number) in hot_pages:
        score += 1.5
    if caption:
        score += 0.3
    score += min(image.width * image.height / 1e6, 0.5)
    return score


def find_visual(query: str, plan: QueryPlan,
                retrieval: RetrievalResult | None = None) -> VisualAnswer:
    """Locate the most relevant visual, rendering a page if nothing was extracted."""
    registry = get_registry()
    results = list(retrieval.results) if retrieval else []
    hot_pages = {(r.document, r.page) for r in results}
    terms = set(keywords(query, 10))

    # 1. an explicit page reference wins outright
    if plan.page_hint:
        document_name = results[0].document if results else None
        if not document_name:
            docs = registry.all()
            document_name = docs[0].name if len(docs) == 1 else None
        if document_name:
            rendered = _render(document_name, plan.page_hint)
            if rendered:
                return rendered

    # 2. an extracted figure whose caption matches
    images = registry.images()
    if images:
        scored = sorted(
            ((_score_image(img, plan.figure_hint, terms, hot_pages), img) for img in images),
            key=lambda t: -t[0],
        )
        floor = 1.0 if plan.figure_hint else 0.3
        best_score, best = scored[0]
        if best_score >= floor:
            return VisualAnswer(
                found=True, image_path=best.path, document=best.document_name,
                page=best.page_number, caption=best.caption,
            )

    # 3. a retrieved chunk that carries its own picture
    for result in results:
        if result.image_path and Path(result.image_path).exists():
            return VisualAnswer(
                found=True, image_path=result.image_path, document=result.document,
                page=result.page, caption="",
            )

    # 4. fall back to rendering the best-matching page
    for result in results[:3]:
        rendered = _render(result.document, result.page)
        if rendered:
            return rendered

    return VisualAnswer(found=False, message="No visual could be located for this question.")


def _render(document_name: str, page_number: int) -> VisualAnswer | None:
    path = _document_path(document_name)
    if not path:
        return None
    try:
        image_path = render_page_image(path, page_number)
    except PDFExtractionError as exc:
        log.warning("page render failed for %s p%d: %s", document_name, page_number, exc)
        return None
    return VisualAnswer(
        found=True, image_path=str(image_path), document=document_name,
        page=page_number, is_page_render=True,
    )


# ---------------------------------------------------------------------------
def analyze_visual(visual: VisualAnswer, question: str, context: str = "",
                   language: str = "en") -> VisualAnswer:
    """Send the located visual to Gemini. Errors land in `message`, not an exception."""
    if not visual.found or not visual.image_path:
        return visual

    template = VISION_PROMPT.get(language, VISION_PROMPT["en"])
    prompt = template.format(
        question=question,
        context=truncate(context, 3000) or "(no surrounding text was retrieved)",
    )
    try:
        visual.analysis = get_gemini().analyze_image(
            visual.image_path, prompt, language=language
        )
        log.info("Gemini analysed %s p%d", visual.document, visual.page)
    except GeminiError as exc:
        visual.message = str(exc)
        log.error("visual analysis failed: %s", exc)
    return visual


def answer_visual_question(query: str, plan: QueryPlan,
                           retrieval: RetrievalResult | None = None,
                           language: str = "en") -> VisualAnswer:
    """find_visual + analyze_visual - the whole visual path in one call."""
    visual = find_visual(query, plan, retrieval)
    if not visual.found:
        return visual

    context = "\n\n".join(
        r.document + " p." + str(r.page) + ": " + r.content
        for r in (retrieval.results[:3] if retrieval else [])
    )
    return analyze_visual(visual, query, context, language)
