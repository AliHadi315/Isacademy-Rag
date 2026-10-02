"""Streamlit application - bilingual (English / العربية).

Run:  streamlit run app/ui/streamlit_app.py     (or: python run.py)

Six pages: Dashboard, Documents, Ask Questions, Data Analysis, Reports,
Settings. Every string comes from app/i18n.py, so switching language switches
the whole interface, including error messages and PCA labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.config import settings  # noqa: E402
from app.document_processing.metadata import get_registry  # noqa: E402
from app.i18n import LANGUAGES, RTL_CSS, is_rtl, t  # noqa: E402
from app.utils.logging import get_logger  # noqa: E402

log = get_logger("ui")

BASE_CSS = """
<style>
  .block-container { padding-top: 2rem; max-width: 1160px; }
  .isa-brand { font-size: 1.05rem; font-weight: 700; line-height: 1.3; }
  .isa-brand span { display:block; font-size:.7rem; font-weight:500; color:#64748b;
                    letter-spacing:.08em; text-transform:uppercase; }
  .isa-card { background:#f1f5f9; color:#0f172a; border:1px solid #cbd5e1;
              border-radius:10px; padding:14px 16px; margin-bottom:10px; }
  .isa-quote { background:#f1f5f9; color:#1e293b; border-inline-start:3px solid #2563eb;
               border-radius:6px; padding:10px 14px; margin:8px 0; font-size:.9rem; }
  .isa-src { font-size:.8rem; color:#64748b; }
  .isa-pill { display:inline-block; padding:2px 10px; border-radius:999px;
              font-size:.72rem; font-weight:600; background:#dbeafe; color:#1e40af; }
  footer, #MainMenu { visibility: hidden; }
</style>
"""


def _lang() -> str:
    return st.session_state.get("lang", settings.default_language)


def _init_state() -> None:
    st.session_state.setdefault("lang", settings.default_language)
    st.session_state.setdefault("page", "nav_dashboard")
    st.session_state.setdefault("last_answer", None)
    st.session_state.setdefault("prefill", "")

    # Streamlit forbids writing a widget-keyed value after that widget has been
    # created, so pages request navigation through this slot instead and it is
    # applied here, before the sidebar radio is built.
    pending = st.session_state.pop("_goto", None)
    if pending in PAGES:
        st.session_state.page = pending


@st.cache_resource(show_spinner=False)
def _warm_up() -> dict:
    """Load the embedding model once, visibly (the first run may download it)."""
    from app.rag.embeddings import get_embedding_provider
    from app.rag.vector_store import get_vector_store

    with st.spinner("Loading the embedding model..."):
        provider = get_embedding_provider()
        return {"embeddings": provider.name, "vectors": get_vector_store().count()}


PAGES = [
    "nav_dashboard", "nav_documents", "nav_ask",
    "nav_data", "nav_reports", "nav_settings",
]


def _sidebar(lang: str, stats: dict) -> str:
    with st.sidebar:
        st.markdown(
            "<div class='isa-brand'>" + t("app_title", lang)
            + "<span>" + t("app_subtitle", lang) + "</span></div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # key="lang" for the same reason as the page radio below: the widget
        # owns st.session_state.lang, so there is only one source of truth.
        chosen = st.selectbox(
            t("language", lang), list(LANGUAGES), key="lang",
            format_func=lambda c: LANGUAGES[c],
        )
        if chosen != lang:
            # the page body above was already rendered in the old language
            st.rerun()

        st.divider()
        # key="page" makes the widget own st.session_state.page. Without it the
        # radio's remembered value silently overwrites any page set elsewhere.
        page = st.radio(
            "nav", PAGES, key="page",
            format_func=lambda key: t(key, lang),
            label_visibility="collapsed",
        )

        st.divider()
        left, right = st.columns(2)
        left.metric(t("documents", lang), stats["documents"])
        right.metric(t("chunks", lang), stats["chunks"])

        if not settings.gemini_configured:
            st.warning(t("no_api_key", lang), icon="⚠️")
    return page


def main() -> None:
    st.set_page_config(
        page_title="Isacademy AI", page_icon="📄",
        layout="wide", initial_sidebar_state="expanded",
    )
    _init_state()
    lang = _lang()

    st.markdown(BASE_CSS, unsafe_allow_html=True)
    if is_rtl(lang):
        st.markdown(RTL_CSS, unsafe_allow_html=True)

    _warm_up()
    stats = get_registry().stats()
    page = _sidebar(lang, stats)

    from app.ui import pages

    renderers = {
        "nav_dashboard": pages.dashboard,
        "nav_documents": pages.documents,
        "nav_ask": pages.ask,
        "nav_data": pages.data_analysis,
        "nav_reports": pages.reports,
        "nav_settings": pages.settings_page,
    }
    try:
        renderers[page](lang)
    except Exception as exc:              # a page error must not blank the app
        log.exception("page %s failed", page)
        st.error(str(exc))
        with st.expander("Details"):
            import traceback

            st.code(traceback.format_exc())


main()
