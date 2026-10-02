"""The question pipeline - one function, the whole workflow.

    question
      -> Query Agent        (what kind of question is this?)
      -> Retrieval Agent    (which passages?)
      -> Visual path        (locate the image, Gemini analyses it)   [if visual]
      -> Semantic PCA       (where does the question sit? what is close?)
      -> Analysis Agent     (Gemini writes the grounded answer)
      -> AnswerResult

Deliberately a plain function rather than an agent framework: the whole flow
fits on one screen and can be explained in a minute.
"""
from __future__ import annotations

import time

from app.agents.analysis_agent import AnalysisAgent
from app.agents.query_agent import QueryAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.analytics.semantic_pca import build_semantic_pca
from app.document_processing.metadata import get_registry
from app.i18n import t
from app.models.schemas import AnswerResult, Intent
from app.rag.embeddings import get_embedding_provider
from app.utils.logging import get_logger
from app.vision.gemini_vision import answer_visual_question

log = get_logger("pipeline")


def ask(question: str, language: str = "en", document_ids: list | None = None,
        with_pca: bool = True) -> AnswerResult:
    """Answer one question end to end. Never raises."""
    started = time.perf_counter()
    registry = get_registry()
    stats = registry.stats()

    result = AnswerResult(query=question, language=language)

    # 1 --- understand -------------------------------------------------
    plan = QueryAgent().run(
        question,
        has_images=stats["images"] > 0 or stats["documents"] > 0,
        has_datasets=stats["datasets"] > 0,
    )
    result.plan = plan

    if not question.strip():
        result.answer = t("no_evidence", language)
        result.grounded = False
        return result

    # 2 --- retrieve ----------------------------------------------------
    retrieval = RetrievalAgent().run(plan.search_query, document_ids=document_ids)
    result.retrieval = retrieval

    if not retrieval.sufficient and "no documents are indexed" in retrieval.message.lower():
        result.answer = t("empty_index", language)
        result.grounded = False
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # 3 --- visual path -------------------------------------------------
    if plan.needs_visual or plan.intent in (Intent.VISUAL, Intent.MIXED):
        try:
            result.visual = answer_visual_question(question, plan, retrieval, language)
        except Exception as exc:
            log.exception("visual path failed")
            result.error = str(exc)

    # 4 --- semantic PCA (attempted for every question) -----------------
    if with_pca:
        try:
            vector = get_embedding_provider().embed_query(plan.search_query)
            result.pca = build_semantic_pca(
                question, vector, language=language, document_ids=document_ids
            )
        except Exception as exc:
            log.warning("semantic PCA unavailable: %s", exc)

    # 5 --- answer -------------------------------------------------------
    answer, findings, error = AnalysisAgent().run(
        question, retrieval, result.visual, language
    )
    result.answer = answer
    result.key_findings = findings
    if error:
        result.error = error
        result.grounded = False
        if not result.answer:
            result.answer = t("gemini_failed", language) + ": " + error
    elif not retrieval.sufficient and not (result.visual and result.visual.found):
        result.grounded = False

    registry.bump("questions")
    result.duration_ms = int((time.perf_counter() - started) * 1000)
    log.info("question answered in %d ms", result.duration_ms)
    return result
