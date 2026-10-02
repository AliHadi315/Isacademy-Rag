"""Analysis Agent - Gemini writes the answer, from retrieved evidence only.

The prompt carries numbered passages with their document and page. Gemini is
told to use nothing else, and to say plainly when the evidence is insufficient.
The citations shown in the UI are built in Python from the retrieved results,
so a page number can never be invented by the model.
"""
from __future__ import annotations

import re

from app.i18n import t
from app.models.gemini import GeminiError, get_gemini
from app.models.schemas import RetrievalResult, VisualAnswer
from app.utils.logging import get_logger

log = get_logger("agent.analysis")

ANSWER_PROMPT = {
    "en": """Answer the user's question using ONLY the document context below.

USER QUESTION:
{question}

DOCUMENT CONTEXT (each block is a retrieved passage with its source):
{context}
{visual}
--- END OF CONTEXT ---

Rules:
- Use only the context above. Do not add outside knowledge.
- Do not invent numbers, sources or page numbers.
- If the context does not answer the question, say exactly that.
- Refer to sources naturally, e.g. "the report states...".
- Be concise: a short accurate answer beats a long vague one.

Write a direct answer, then a short "Key findings" list of 2-5 bullet points.""",
    "ar": """أجب عن سؤال المستخدم باستخدام سياق المستندات أدناه فقط.

سؤال المستخدم:
{question}

سياق المستندات (كل مقطع هو نص مسترجَع مع مصدره):
{context}
{visual}
--- نهاية السياق ---

القواعد:
- استخدم السياق أعلاه فقط ولا تضف معرفة خارجية.
- لا تخترع أرقاماً أو مصادر أو أرقام صفحات.
- إذا كان السياق لا يجيب عن السؤال، قل ذلك صراحةً.
- أشر إلى المصادر بشكل طبيعي، مثل "يذكر التقرير...".
- كن موجزاً: إجابة قصيرة دقيقة أفضل من إجابة طويلة غامضة.

اكتب إجابة مباشرة، ثم قائمة قصيرة بعنوان "أهم النتائج" من 2 إلى 5 نقاط.""",
}

VISUAL_BLOCK = {
    "en": "\n\nVISUAL EVIDENCE (Gemini's analysis of {source}):\n{analysis}\n",
    "ar": "\n\nدليل مرئي (تحليل Gemini لـ {source}):\n{analysis}\n",
}

FINDINGS_HEADERS = ("key findings", "أهم النتائج", "النتائج الرئيسية")


class AnalysisAgent:
    def __init__(self, gemini=None):
        self.gemini = gemini

    def run(self, question: str, retrieval: RetrievalResult | None = None,
            visual: VisualAnswer | None = None, language: str = "en") -> tuple:
        """Returns (answer_text, key_findings, error). Never raises."""
        from app.agents.retrieval_agent import RetrievalAgent

        has_text = bool(retrieval and retrieval.results)
        has_visual = bool(visual and visual.found and visual.analysis)

        if not has_text and not has_visual:
            log.info("declined: no evidence")
            return t("no_evidence", language), [], ""

        context = (
            RetrievalAgent.context_block(retrieval.results)
            if has_text else "(no text passages were retrieved)"
        )
        visual_block = ""
        if has_visual:
            visual_block = VISUAL_BLOCK.get(language, VISUAL_BLOCK["en"]).format(
                source=visual.document + " page " + str(visual.page),
                analysis=visual.analysis,
            )

        prompt = ANSWER_PROMPT.get(language, ANSWER_PROMPT["en"]).format(
            question=question, context=context, visual=visual_block,
        )

        try:
            raw = (self.gemini or get_gemini()).generate(prompt, language=language)
        except GeminiError as exc:
            log.error("answer generation failed: %s", exc)
            return "", [], str(exc)

        answer, findings = self._split(raw)
        log.info("answer generated (%d chars, %d findings)", len(answer), len(findings))
        return answer, findings, ""

    # ------------------------------------------------------------------
    @staticmethod
    def _split(raw: str) -> tuple:
        """Separate the prose answer from the trailing bullet list."""
        lines = (raw or "").strip().splitlines()
        body, findings, in_findings = [], [], False

        for line in lines:
            stripped = line.strip()
            lowered = stripped.lower().strip("*#: ")
            if any(lowered.startswith(h) for h in FINDINGS_HEADERS):
                in_findings = True
                continue
            if in_findings:
                # Strip only the list marker. A blanket lstrip("-*• ") would
                # eat the opening ** of "- **Bold:** text" and leave the
                # closing one stranded in the UI.
                cleaned = re.sub(r"^(?:[-*••]|\d+[.)])\s+", "", stripped).strip()
                if cleaned:
                    findings.append(cleaned)
            else:
                body.append(line)

        return "\n".join(body).strip() or (raw or "").strip(), findings[:6]
