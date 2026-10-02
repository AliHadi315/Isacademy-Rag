"""Query agent, Gemini client behaviour, the pipeline, i18n and reports."""
from pathlib import Path

import pytest

from app.agents.analysis_agent import AnalysisAgent
from app.agents.query_agent import QueryAgent
from app.i18n import LANGUAGES, STRINGS, is_rtl, t
from app.models.gemini import GeminiClient, GeminiError
from app.models.schemas import (
    AnswerResult, Chunk, ContentType, Intent, RetrievalResult, SearchResult,
)
from app.rag.embeddings import get_embedding_provider


# ------------------------------------------------------------- query agent
@pytest.mark.parametrize("query,intent", [
    ("What are the main findings?", Intent.RAG),
    ("What does Figure 3 show?", Intent.VISUAL),
    ("Which column has the highest average?", Intent.DATA),
    ("Write a report about the findings", Intent.REPORT),
])
def test_query_agent_routes(query, intent):
    assert QueryAgent().run(query).intent is intent


def test_query_agent_extracts_a_figure_reference():
    plan = QueryAgent().run("What does Figure 3 show?")
    assert plan.figure_hint == "3" and plan.needs_visual


def test_query_agent_extracts_a_page_reference():
    plan = QueryAgent().run("Explain the graph on page 15.")
    assert plan.page_hint == 15 and plan.needs_visual


def test_query_agent_understands_arabic():
    plan = QueryAgent().run("ماذا يعرض الشكل 2؟")
    assert plan.needs_visual and plan.figure_hint == "2"


def test_query_agent_strips_conversational_prefix():
    plan = QueryAgent().run("Can you please tell me the main findings?")
    assert not plan.search_query.lower().startswith("can you")


def test_query_agent_always_keeps_text_retrieval():
    """Even a pure visual question needs page text for context and citation."""
    assert QueryAgent().run("What does Figure 1 show?").needs_text


def test_query_agent_wont_route_to_missing_capability():
    plan = QueryAgent().run("Which column correlates most?", has_datasets=False)
    assert not plan.needs_data


def test_query_agent_handles_empty():
    assert not QueryAgent().run("").needs_text


# ------------------------------------------------------------- Gemini client
def test_gemini_without_a_key_raises_a_clear_error():
    client = GeminiClient(api_key="", model="gemini-2.0-flash")
    assert not client.configured
    with pytest.raises(GeminiError) as exc:
        client.generate("hello")
    assert "GEMINI_API_KEY" in str(exc.value)


def test_gemini_vision_without_a_key_raises():
    client = GeminiClient(api_key="", model="gemini-2.0-flash")
    with pytest.raises(GeminiError):
        client.analyze_image(__file__, "describe this")


def test_gemini_reports_a_missing_image(tmp_path):
    client = GeminiClient(api_key="fake-key-for-shape-check")
    with pytest.raises(GeminiError) as exc:
        client.analyze_image(tmp_path / "nope.png", "describe")
    assert "not found" in str(exc.value).lower()


# ----------------------------------------------------------- analysis agent
def _retrieval():
    return RetrievalResult(
        query="accuracy", sufficient=True,
        results=[SearchResult(content="Accuracy improved to 88.4 percent.",
                              document="paper.pdf", page=5, chunk_id="c1", score=0.8)],
    )


def test_analysis_declines_without_evidence():
    answer, findings, error = AnalysisAgent().run("anything", RetrievalResult())
    assert not error
    assert "not find enough information" in answer.lower()


def test_analysis_declines_in_arabic():
    answer, _, _ = AnalysisAgent().run("أي شيء", RetrievalResult(), language="ar")
    assert "لم أجد معلومات كافية" in answer


def test_analysis_surfaces_a_gemini_failure():
    answer, findings, error = AnalysisAgent().run("What happened?", _retrieval())
    assert error and "GEMINI_API_KEY" in error
    assert answer == ""          # no fabricated answer


def test_analysis_splits_findings_from_prose():
    body, findings = AnalysisAgent._split(
        "The study reports a gain.\n\nKey findings:\n- Accuracy rose\n- Latency fell"
    )
    assert "gain" in body
    assert findings == ["Accuracy rose", "Latency fell"]


# ----------------------------------------------------------------- pipeline
def test_pipeline_reports_an_empty_index(clean_index):
    from app.agents.pipeline import ask

    result = ask("What are the main findings?", language="en")
    assert not result.grounded
    assert "no documents" in result.answer.lower()


def test_pipeline_runs_every_stage(clean_index):
    from app.agents.pipeline import ask

    provider = get_embedding_provider()
    texts = [
        "Principal component analysis retained 89 percent of the variance.",
        "Validation accuracy improved to 88.4 percent after fine-tuning.",
        "Serving costs fell 28 percent quarter over quarter.",
        "Cold starts remain the dominant latency risk in production.",
        "The robustness suite showed no regression on held-out data.",
    ]
    chunks = [
        Chunk(chunk_id="c" + str(i), document_id="d1", document_name="paper.pdf",
              page_number=i + 1, content=text, content_type=ContentType.TEXT,
              char_count=len(text), chunk_index=i)
        for i, text in enumerate(texts)
    ]
    clean_index.add(chunks, provider.embed_documents(texts))

    result = ask("What happened to accuracy?", language="en")
    assert result.plan is not None
    assert result.retrieval is not None and result.retrieval.results
    assert result.pca is not None and result.pca.available   # PCA on every question
    assert result.sources                                    # real citations
    assert all(" — Page " in s for s in result.sources)
    # no API key: Gemini failure is reported, never papered over
    assert result.error and not result.grounded


# --------------------------------------------------------------------- i18n
def test_every_string_has_both_languages():
    for key, entry in STRINGS.items():
        assert "en" in entry and entry["en"], key
        assert "ar" in entry and entry["ar"], key


def test_translation_and_fallback():
    assert t("answer", "en") == "Answer"
    assert t("answer", "ar") == "الإجابة"
    assert t("does_not_exist", "en") == "does_not_exist"


def test_rtl_only_for_arabic():
    assert is_rtl("ar") and not is_rtl("en")
    assert set(LANGUAGES) == {"en", "ar"}


# ------------------------------------------------------------------ reports
def _answer() -> AnswerResult:
    return AnswerResult(
        query="What are the main findings?", language="en",
        answer="Accuracy improved to 88.4 percent.",
        key_findings=["Accuracy rose 17.2 points."],
        retrieval=_retrieval(),
    )


def test_report_sections():
    from app.reports.generator import build_sections

    titles = [s[0] for s in build_sections(_answer(), "en") if s[0]]
    assert "Answer" in titles and "Sources" in titles


def test_markdown_report():
    from app.reports.generator import to_markdown

    text = to_markdown(_answer(), "en")
    assert "88.4" in text and "paper.pdf" in text


def test_arabic_report_is_arabic():
    from app.reports.generator import to_markdown

    result = _answer()
    result.language = "ar"
    text = to_markdown(result, "ar")
    assert "الإجابة" in text and "المصادر" in text


def test_html_escapes_injection():
    from app.reports.generator import to_html

    result = _answer()
    result.answer = "<script>alert('x')</script>"
    assert "<script>" not in to_html(result, "en")


def test_html_marks_rtl_for_arabic():
    from app.reports.generator import to_html

    assert "dir='rtl'" in to_html(_answer(), "ar")


def test_render_writes_files():
    from app.reports.generator import render

    files = render(_answer(), ["markdown", "html"], "en")
    assert Path(files["markdown"]).exists()
    assert Path(files["html"]).exists()


def test_pdf_export():
    pytest.importorskip("reportlab")
    from app.reports.generator import render

    files = render(_answer(), ["pdf"], "en")
    assert "pdf" in files, files
    path = Path(files["pdf"])
    assert path.exists() and path.read_bytes()[:4] == b"%PDF"


# ------------------------------------------------------------------- arabic
def test_arabic_detection():
    from app.reports.arabic import has_arabic

    assert has_arabic("ما هي أهم النتائج؟")
    assert not has_arabic("What are the main findings?")


def test_shaping_leaves_latin_untouched():
    from app.reports.arabic import shape

    assert shape("Hello world 123") == "Hello world 123"
    assert shape("") == ""


def test_shaping_changes_arabic():
    """Reshaping joins the letters, so the output differs from the input."""
    from app.reports.arabic import shape

    raw = "ما هي أهم النتائج"
    shaped = shape(raw)
    assert shaped and shaped != raw


def test_arabic_pdf_report_contains_arabic_glyphs(tmp_path):
    """The regression this guards: Arabic used to be silently dropped."""
    pytest.importorskip("reportlab")
    pymupdf = pytest.importorskip("pymupdf")
    import re

    from app.reports.generator import render

    result = _answer()
    result.language = "ar"
    result.query = "ما هي أهم النتائج؟"
    result.answer = "تحسّنت دقة التحقق بعد الضبط الدقيق."
    files = render(result, ["pdf"], "ar")
    assert "pdf" in files, files

    doc = pymupdf.open(files["pdf"])
    text = "".join(doc.load_page(i).get_text() for i in range(doc.page_count))
    assert re.search(r"[\u0600-\u06FF\uFE70-\uFEFF]", text), \
        "no Arabic glyphs reached the PDF"
