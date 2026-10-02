"""Query Agent - what kind of question is this?

Decides whether the question needs text RAG, a visual, dataset numbers, or a
report, and pulls out any explicit "Figure 3" / "page 15" reference.

Keyword rules do the routing. They are instant, free, work in Arabic and
English, and are trivial to explain in a presentation - which matters more here
than squeezing out the last few percent of classification accuracy. Gemini is
saved for the work that actually needs a language model: answering.
"""
from __future__ import annotations

import re

from app.models.schemas import Intent, QueryPlan
from app.utils.logging import get_logger

log = get_logger("agent.query")

# English + Arabic cues
VISUAL_PAT = re.compile(
    r"\b(figure|fig\.?|chart|graph|plot|image|picture|diagram|screenshot|visual|"
    r"infographic|axis|axes|legend)\b"
    r"|(شكل|رسم|مخطط|صورة|رسمة|بياني|منحنى)",
    re.I | re.UNICODE,
)
TABLE_PAT = re.compile(r"\b(table)\b|(جدول)", re.I | re.UNICODE)
DATA_PAT = re.compile(
    r"\b(dataset|data\s*set|csv|excel|spreadsheet|column|correlat\w*|mean|median|"
    r"average|distribution|histogram|scatter|variance|outlier|highest|lowest|pca)\b"
    r"|(بيانات|عمود|أعمدة|ارتباط|متوسط|توزيع|أعلى قيمة|أدنى قيمة|انحراف)",
    re.I | re.UNICODE,
)
REPORT_PAT = re.compile(
    r"\b(report|publish|export|write\s*up|deliverable)\b|(تقرير|تقريرا|اكتب تقرير)",
    re.I | re.UNICODE,
)
FIGURE_REF = re.compile(
    r"(?:fig(?:ure)?|chart|exhibit|diagram|table|شكل|جدول|مخطط)\.?\s*(\d+[a-z]?)",
    re.I | re.UNICODE,
)
PAGE_REF = re.compile(r"(?:page|صفحة|ص\.?)\s*(\d{1,4})", re.I | re.UNICODE)

CONVERSATIONAL = re.compile(
    r"^\s*(can you|could you|please|tell me|i want to know|what about|"
    r"من فضلك|هل يمكنك|أخبرني|ما رأيك)\b[,\s]*",
    re.I | re.UNICODE,
)


class QueryAgent:
    def run(self, query: str, has_images: bool = True,
            has_datasets: bool = True) -> QueryPlan:
        query = (query or "").strip()
        if not query:
            return QueryPlan(original_query="", needs_text=False,
                             reason="Empty question.")

        wants_visual = bool(VISUAL_PAT.search(query) or TABLE_PAT.search(query))
        wants_data = bool(DATA_PAT.search(query))
        wants_report = bool(REPORT_PAT.search(query))

        figure_match = FIGURE_REF.search(query)
        page_match = PAGE_REF.search(query)
        figure_hint = figure_match.group(1) if figure_match else ""
        page_hint = int(page_match.group(1)) if page_match else 0

        # an explicit "Figure 3" or "page 12" is a visual question even if no
        # other visual word appears
        if figure_hint or page_hint:
            wants_visual = True

        if not has_images:
            wants_visual = False
        if not has_datasets:
            wants_data = False

        if wants_visual and wants_data:
            intent = Intent.MIXED
        elif wants_visual:
            intent = Intent.MIXED if not figure_hint else Intent.VISUAL
        elif wants_report:
            intent = Intent.REPORT
        elif wants_data:
            intent = Intent.DATA
        else:
            intent = Intent.RAG

        cues = [
            name for name, on in [
                ("visual", wants_visual), ("data", wants_data), ("report", wants_report)
            ] if on
        ]
        plan = QueryPlan(
            original_query=query,
            intent=intent,
            search_query=self._clean(query),
            # text retrieval always runs: even a pure visual question needs the
            # surrounding page text for context and citation
            needs_text=True,
            needs_visual=wants_visual,
            needs_data=wants_data,
            wants_report=wants_report,
            figure_hint=figure_hint,
            page_hint=page_hint,
            reason="cues: " + (", ".join(cues) or "plain text question"),
        )
        log.info(
            "Query agent: intent=%s visual=%s data=%s figure=%r page=%s",
            plan.intent.value, plan.needs_visual, plan.needs_data,
            plan.figure_hint, plan.page_hint or "-",
        )
        return plan

    @staticmethod
    def _clean(query: str) -> str:
        """Drop conversational scaffolding that only pollutes the embedding."""
        cleaned = CONVERSATIONAL.sub("", query)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.!؟")
        return cleaned or query
