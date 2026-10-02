"""UI tests driven through Streamlit's own headless runner.

`AppTest` executes the real app script - the same navigation, language
switching and page rendering a user drives with the mouse - so these cover the
interaction logic without needing a browser.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "ui" / "streamlit_app.py")
TIMEOUT = 90


def _app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    return app.run()


def _text(app) -> str:
    """All rendered text on the page, for substring assertions."""
    parts = []
    for collection in ("title", "header", "subheader", "markdown", "caption",
                       "info", "warning", "error", "success", "text"):
        for element in getattr(app, collection, []):
            parts.append(str(getattr(element, "value", "")))
    for element in app.sidebar.markdown:
        parts.append(str(element.value))
    return "\n".join(parts)


# --------------------------------------------------------------------- boot
def test_app_starts_without_exception():
    app = _app()
    assert not app.exception


def test_dashboard_is_the_landing_page():
    app = _app()
    assert "Dashboard" in _text(app)
    assert app.session_state["page"] == "nav_dashboard"


def test_dashboard_has_no_system_status_notes_or_activity():
    """The brief explicitly removes these from the Dashboard."""
    text = _text(_app()).lower()
    assert "system status" not in text
    assert "recent activity" not in text
    assert "notes" not in text


def test_dashboard_shows_the_three_metrics():
    app = _app()
    labels = {m.label for m in app.metric}
    assert {"Documents", "Chunks", "Questions Answered"} <= labels


def test_missing_api_key_is_reported():
    app = _app()
    warnings = " ".join(str(w.value) for w in app.sidebar.warning)
    assert "GEMINI_API_KEY" in warnings


# --------------------------------------------------------------- navigation
@pytest.mark.parametrize("page,expected", [
    ("nav_documents", "Documents"),
    ("nav_ask", "Ask a Question"),
    ("nav_data", "Data Analysis"),
    ("nav_reports", "Reports"),
    ("nav_settings", "Settings"),
])
def test_every_page_renders(page, expected):
    app = _app()
    app.session_state["page"] = page
    app.run()
    assert not app.exception, page
    assert expected in _text(app)


def test_sidebar_radio_changes_the_page():
    app = _app()
    app.sidebar.radio[0].set_value("nav_ask").run()
    assert not app.exception
    assert app.session_state["page"] == "nav_ask"
    assert "Ask a Question" in _text(app)


def test_quick_action_button_navigates():
    """The Quick Action buttons must actually move the user to that page."""
    app = _app()
    ask_button = next(b for b in app.button if b.label == "Ask a Question")
    ask_button.click().run()
    assert not app.exception
    assert app.session_state["page"] == "nav_ask"


# ----------------------------------------------------------------- language
def test_language_switches_the_whole_interface():
    app = _app()
    app.sidebar.selectbox[0].set_value("ar").run()
    assert not app.exception
    assert app.session_state["lang"] == "ar"

    text = _text(app)
    assert "لوحة التحكم" in text          # Dashboard
    assert "إجراءات سريعة" in text        # Quick Actions


def test_error_messages_are_translated():
    app = _app()
    app.sidebar.selectbox[0].set_value("ar").run()
    warnings = " ".join(str(w.value) for w in app.sidebar.warning)
    assert "مفتاح Gemini" in warnings


def test_arabic_pages_render():
    app = _app()
    app.sidebar.selectbox[0].set_value("ar").run()
    for page, expected in [("nav_ask", "اطرح سؤالاً"),
                           ("nav_data", "تحليل البيانات"),
                           ("nav_reports", "التقارير")]:
        app.session_state["page"] = page
        app.run()
        assert not app.exception, page
        assert expected in _text(app), page


def test_language_persists_across_navigation():
    app = _app()
    app.sidebar.selectbox[0].set_value("ar").run()
    app.session_state["page"] = "nav_settings"
    app.run()
    assert app.session_state["lang"] == "ar"
    assert "الإعدادات" in _text(app)


# --------------------------------------------------------------- ask page
def test_ask_page_without_documents_explains_why(clean_index):
    app = _app()
    app.session_state["page"] = "nav_ask"
    app.run()
    assert not app.exception
    assert "No documents are indexed" in _text(app)


def test_reports_page_without_an_answer_explains_why():
    app = _app()
    app.session_state["page"] = "nav_reports"
    app.run()
    assert not app.exception
    assert "Ask a question first" in _text(app)


def test_settings_never_shows_the_api_key():
    app = _app()
    app.session_state["page"] = "nav_settings"
    app.run()
    assert not app.exception
    rendered = _text(app) + "".join(str(j.value) for j in app.json)
    assert "not set" in rendered or "chars" in rendered


# ------------------------------------------------- a real answer, rendered
def _indexed_answer(clean_index, language="en"):
    """Index a small corpus, ask a question, return the AnswerResult."""
    from app.agents.pipeline import ask
    from app.document_processing.metadata import get_registry
    from app.models.schemas import Chunk, ContentType, Document
    from app.rag.embeddings import get_embedding_provider

    provider = get_embedding_provider()
    texts = [
        "Principal component analysis retained 89 percent of the variance.",
        "Validation accuracy improved to 88.4 percent after fine-tuning.",
        "Serving costs fell 28 percent quarter over quarter.",
        "Cold starts remain the dominant latency risk in production.",
        "The robustness suite showed no regression on held-out data.",
    ]
    chunks = [
        Chunk(chunk_id="u" + str(i), document_id="d1", document_name="paper.pdf",
              page_number=i + 1, content=text, content_type=ContentType.TEXT,
              char_count=len(text), chunk_index=i)
        for i, text in enumerate(texts)
    ]
    clean_index.add(chunks, provider.embed_documents(texts))
    get_registry().add(Document(
        document_id="d1", name="paper.pdf", path="paper.pdf", file_type="pdf",
        page_count=len(texts), chunk_count=len(chunks),
    ))
    return ask("What happened to accuracy?", language=language)


def test_ask_page_renders_answer_sources_and_pca(clean_index):
    result = _indexed_answer(clean_index)
    assert result.pca.available

    app = _app()
    app.session_state["page"] = "nav_ask"
    app.session_state["last_answer"] = result
    app.run()

    assert not app.exception
    text = _text(app)
    assert "Answer" in text
    assert "Sources" in text
    assert "paper.pdf" in text                 # a real citation
    assert "Semantic PCA Map" in text
    assert "Query Cluster" in text
    assert "Closest cluster" in text
    assert "Why Are They Close?" in text
    assert "PCA is a 2D projection" in text    # the honesty note
    # the interactive map itself was rendered
    assert len(app.get("plotly_chart")) >= 1


def test_ask_page_pca_labels_are_arabic(clean_index):
    result = _indexed_answer(clean_index, language="ar")

    app = _app()
    app.sidebar.selectbox[0].set_value("ar").run()
    app.session_state["page"] = "nav_ask"
    app.session_state["last_answer"] = result
    app.run()

    assert not app.exception
    text = _text(app)
    assert "خريطة PCA الدلالية" in text        # Semantic PCA Map
    assert "مجموعة السؤال" in text             # Query Cluster
    assert "لماذا هذه العناصر قريبة؟" in text   # Why are they close?


def test_reports_page_builds_from_the_answer(clean_index):
    result = _indexed_answer(clean_index)

    app = _app()
    app.session_state["page"] = "nav_reports"
    app.session_state["last_answer"] = result
    app.run()

    assert not app.exception
    assert "Generate report" in " ".join(b.label for b in app.button)
