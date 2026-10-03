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

# Palette and spacing live as CSS custom properties so there is one place to
# change them. The colours mirror .streamlit/config.toml, which themes the
# widgets Streamlit renders itself.
BASE_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');

  :root {
    --isa-primary:   #1E40AF;
    --isa-primary-2: #3B82F6;
    --isa-accent:    #F59E0B;
    --isa-ink:       #0F172A;
    --isa-muted:     #475569;
    --isa-subtle:    #64748B;
    --isa-line:      #E2E8F0;
    --isa-surface:   #F8FAFC;
    --isa-surface-2: #F1F5F9;
    --isa-radius:    0.625rem;
    --isa-gap:       1rem;
    --isa-shadow:    0 1px 2px rgba(15,23,42,.04), 0 1px 3px rgba(15,23,42,.06);
    --isa-shadow-lg: 0 4px 6px -1px rgba(15,23,42,.07), 0 2px 4px -2px rgba(15,23,42,.05);
  }

  /* dark mode: only the tokens change, every rule below follows */
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --isa-ink:       #E2E8F0;
      --isa-muted:     #94A3B8;
      --isa-subtle:    #94A3B8;
      --isa-line:      #1E293B;
      --isa-surface:   #0F172A;
      --isa-surface-2: #111A2E;
      --isa-shadow:    0 1px 2px rgba(0,0,0,.4);
      --isa-shadow-lg: 0 4px 10px rgba(0,0,0,.45);
    }
  }

  .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }

  /* ---- typography scale ---- */
  h1, h2, h3 { letter-spacing: -0.018em; }
  h1 { font-weight: 650 !important; }
  h2 { font-weight: 600 !important; margin-top: 0.4em !important; }
  h3 { font-weight: 600 !important; font-size: 1.05rem !important; }
  p, li { line-height: 1.65; }

  /* ---- brand ---- */
  .isa-brand { font-size: 1.1rem; font-weight: 700; line-height: 1.25;
               color: var(--isa-ink); }
  .isa-brand span { display:block; font-size:.66rem; font-weight:500;
                    color: var(--isa-subtle); letter-spacing:.1em;
                    text-transform:uppercase; margin-top:2px; }

  /* ---- surfaces ---- */
  .isa-card { background: var(--isa-surface-2); color: var(--isa-ink);
              border: 1px solid var(--isa-line); border-radius: var(--isa-radius);
              padding: 14px 16px; margin-bottom: 10px; box-shadow: var(--isa-shadow); }

  .isa-quote { background: var(--isa-surface-2); color: var(--isa-ink);
               border-inline-start: 3px solid var(--isa-primary);
               border-radius: var(--isa-radius); padding: 12px 16px; margin: 8px 0;
               font-size: .9rem; line-height: 1.6; }

  /* ---- source list: a real list, not loose grey text ---- */
  .isa-src { font-size: .82rem; color: var(--isa-muted); line-height: 1.75; }
  .isa-source-row { display:flex; gap:.6rem; align-items:baseline;
                    padding:.45rem .7rem; border-radius:.45rem;
                    border:1px solid transparent; transition: background-color .18s ease,
                    border-color .18s ease; }
  .isa-source-row:hover { background: var(--isa-surface-2);
                          border-color: var(--isa-line); }
  .isa-source-n { flex:0 0 auto; font-family:'Fira Code', ui-monospace, monospace;
                  font-size:.72rem; font-weight:500; color: var(--isa-primary);
                  background: var(--isa-surface-2); border-radius:.3rem;
                  padding:.08rem .38rem; }
  .isa-source-doc { color: var(--isa-ink); font-size:.84rem; }
  .isa-source-page { color: var(--isa-subtle); font-size:.78rem;
                     font-family:'Fira Code', ui-monospace, monospace; }

  /* ---- pills ---- */
  .isa-pill { display:inline-block; padding:.18rem .6rem; border-radius:999px;
              font-size:.72rem; font-weight:600; letter-spacing:.01em;
              background:#DBEAFE; color:#1E40AF; }
  .isa-pill-accent { background:#FEF3C7; color:#92400E; }

  /* ---- metrics ---- */
  [data-testid="stMetric"] { background: var(--isa-surface-2);
      border: 1px solid var(--isa-line); border-radius: var(--isa-radius);
      padding: 1rem 1.15rem; box-shadow: var(--isa-shadow); }
  [data-testid="stMetricLabel"] { color: var(--isa-muted) !important;
      font-size:.78rem !important; letter-spacing:.04em; text-transform:uppercase; }

  /* ---- interaction: every clickable thing says so ---- */
  .stButton > button, .stDownloadButton > button,
  [role="radio"], [role="tab"], summary { cursor: pointer; }
  .stButton > button { transition: background-color .18s ease, border-color .18s ease,
                       box-shadow .18s ease; font-weight: 500; }
  .stButton > button:hover { box-shadow: var(--isa-shadow-lg); }
  .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
      outline: 2px solid var(--isa-primary); outline-offset: 2px; }

  /* ---- charts sit on a card like everything else ---- */
  .js-plotly-plot { border: 1px solid var(--isa-line);
      border-radius: var(--isa-radius); padding: .4rem;
      background: var(--isa-surface); box-shadow: var(--isa-shadow); }

  [data-testid="stExpander"] { border-radius: var(--isa-radius) !important;
      border-color: var(--isa-line) !important; }

  /* ---- sidebar navigation ----
     Styled from Streamlit's own radio rather than a custom component: a
     custom one is invisible to AppTest, which would make navigation
     untestable. This keeps the widget real and the tests honest. */
  [data-testid="stSidebar"] [role="radiogroup"] { gap: .15rem; }
  [data-testid="stSidebar"] [role="radiogroup"] > label {
      display: flex; align-items: center; width: 100%;
      padding: .5rem .7rem; margin: 0; border-radius: .5rem;
      border-inline-start: 3px solid transparent;
      color: var(--isa-muted); font-weight: 500; font-size: .92rem;
      cursor: pointer;
      transition: background-color .18s ease, color .18s ease,
                  border-color .18s ease;
  }
  [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
      background: var(--isa-surface-2); color: var(--isa-ink);
  }
  /* Hide the radio dot - the row itself carries the state. Targeted
     structurally (the div holding the label text) because Streamlit's
     emotion class names change between releases. */
  [data-testid="stSidebar"] [role="radiogroup"] label
    div:has(> [data-testid="stMarkdownContainer"]) > div:first-child {
      display: none !important;
  }

  /* the sidebar is narrow: metric cards need tighter type than the main area */
  [data-testid="stSidebar"] [data-testid="stMetric"] { padding: .6rem .7rem; }
  [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
      font-size: .64rem !important; letter-spacing: .02em;
  }
  [data-testid="stSidebar"] [data-testid="stMetricValue"] {
      font-size: 1.35rem !important;
  }
  [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
      background: #DBEAFE; color: var(--isa-primary); font-weight: 600;
      border-inline-start-color: var(--isa-primary);
  }
  [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:focus-visible) {
      outline: 2px solid var(--isa-primary); outline-offset: 1px;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"])
      [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        background: #1E293B; color: #93C5FD; border-inline-start-color: #60A5FA;
    }
  }

  footer, #MainMenu { visibility: hidden; }

  /* ---- respect the OS motion preference ---- */
  @media (prefers-reduced-motion: reduce) {
    * { transition-duration: .01ms !important; animation-duration: .01ms !important; }
  }
</style>
"""

# Arabic needs a family that actually has the glyphs; Fira Sans does not.
ARABIC_FONT_CSS = """
<style>
  html, body, [class*="st-"], .stMarkdown, input, textarea, button, select {
      font-family: 'IBM Plex Sans Arabic', 'Fira Sans', system-ui, sans-serif !important;
  }
  /* numbers, code and file paths stay in the Latin face */
  code, pre, .isa-source-n, .isa-source-page, [data-testid="stMetricValue"] {
      font-family: 'Fira Code', ui-monospace, monospace !important;
  }
</style>
"""


def _lang() -> str:
    return st.session_state.get("lang", settings.default_language)


def _init_state() -> None:
    # A ?lang= / ?q= deep link picks the language and jumps straight to the
    # answer, so a result can be shared as a URL.
    linked_lang = (st.query_params.get("lang") or "").strip().lower()
    st.session_state.setdefault(
        "lang", linked_lang if linked_lang in LANGUAGES else settings.default_language
    )
    st.session_state.setdefault(
        "page", "nav_ask" if st.query_params.get("q") else "nav_dashboard"
    )
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
        st.markdown(ARABIC_FONT_CSS, unsafe_allow_html=True)
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
