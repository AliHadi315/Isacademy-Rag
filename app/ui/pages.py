"""The six pages. Each takes the active language and renders itself."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config import reload_settings, settings
from app.document_processing.metadata import get_registry
from app.document_processing.processor import STAGES, delete_document, index_path, index_upload
from app.i18n import t
from app.utils.file_utils import UnsafeFileError, human_size

ICON = {"done": "✓", "running": "⏳", "skipped": "↷", "pending": "·"}

EXAMPLES = {
    "en": [
        "What are the main findings?",
        "What does Figure 1 show?",
        "Explain the chart on page 2.",
        "Compare the two documents.",
    ],
    "ar": [
        "ما هي أهم النتائج؟",
        "ماذا يعرض الشكل 1؟",
        "اشرح الرسم البياني في صفحة 2.",
        "قارن بين المستندين.",
    ],
}


def _goto(page: str) -> None:
    """Request a page change; streamlit_app applies it before the nav widget."""
    st.session_state["_goto"] = page
    st.rerun()


# ===========================================================================
def dashboard(lang: str) -> None:
    st.title(t("nav_dashboard", lang))
    stats = get_registry().stats()

    columns = st.columns(3)
    columns[0].metric(t("documents", lang), stats["documents"])
    columns[1].metric(t("chunks", lang), f"{stats['chunks']:,}")
    columns[2].metric(t("questions_answered", lang), stats["questions"])

    st.divider()
    st.subheader(t("quick_actions", lang))
    left, mid, right = st.columns(3)
    if left.button(t("upload_document", lang), use_container_width=True):
        _goto("nav_documents")
    if mid.button(t("ask_a_question", lang), use_container_width=True):
        _goto("nav_ask")
    if right.button(t("analyze_data", lang), use_container_width=True):
        _goto("nav_data")

    documents = get_registry().all()
    if documents:
        st.divider()
        st.subheader(t("indexed_documents", lang))
        import pandas as pd

        st.dataframe(
            pd.DataFrame([
                {
                    t("documents", lang): d.name,
                    t("pages", lang): d.page_count,
                    t("chunks", lang): d.chunk_count,
                    t("figures", lang): d.image_count,
                }
                for d in documents
            ]),
            use_container_width=True, hide_index=True,
        )


# ===========================================================================
def documents(lang: str) -> None:
    st.title(t("nav_documents", lang))
    registry = get_registry()

    uploads = st.file_uploader(
        t("upload_pdfs", lang),
        type=[e.lstrip(".") for e in settings.allowed_doc_ext],
        accept_multiple_files=True,
    )
    left, right = st.columns([1, 3])
    reindex = left.checkbox(t("force_reindex", lang), value=False)

    if uploads and right.button(t("process_files", lang), type="primary"):
        for upload in uploads:
            st.markdown("**" + upload.name + "**")
            placeholder = st.empty()
            state = {stage: "pending" for stage in STAGES}

            def progress(stage, status, _s=state, _ph=placeholder):
                _s[stage] = status
                _ph.markdown("  \n".join(
                    ICON.get(_s[k], "·") + " " + k for k in STAGES
                ))

            outcome = index_upload(upload.getvalue(), upload.name,
                                   progress=progress, reindex=reindex)
            if outcome.error:
                st.error(t("extraction_failed", lang) + ": " + outcome.error)
            elif outcome.skipped:
                st.info(t("already_indexed", lang))
            else:
                d = outcome.document
                st.success(
                    t("processing_done", lang) + " — " + str(d.page_count) + " "
                    + t("pages", lang) + ", " + str(d.chunk_count) + " "
                    + t("chunks", lang) + ", " + str(d.image_count) + " "
                    + t("figures", lang)
                )
        st.rerun()

    demo_dir = Path(settings.upload_dir).parent.parent / "demo"
    demo_pdfs = sorted(demo_dir.glob("*.pdf")) if demo_dir.exists() else []
    if demo_pdfs and st.button(t("load_demo", lang)):
        for pdf in demo_pdfs:
            outcome = index_path(pdf)
            if outcome.error:
                st.error(pdf.name + ": " + outcome.error)
            else:
                st.success(pdf.name)
        st.rerun()

    st.divider()
    st.subheader(t("indexed_documents", lang))
    docs = registry.all()
    if not docs:
        st.info(t("no_documents", lang))
        return

    for doc in docs:
        with st.container(border=True):
            head, actions = st.columns([4, 1])
            head.markdown("**" + doc.name + "**")
            head.markdown(
                "<span class='isa-src'>" + str(doc.page_count) + " " + t("pages", lang)
                + " · " + str(doc.chunk_count) + " " + t("chunks", lang)
                + " · " + str(doc.image_count) + " " + t("figures", lang)
                + " · " + str(doc.table_count) + " " + t("tables", lang)
                + " · " + human_size(doc.size_bytes) + "</span>",
                unsafe_allow_html=True,
            )
            if actions.button(t("reindex", lang), key="ri_" + doc.document_id):
                outcome = index_path(Path(doc.path), reindex=True)
                st.error(outcome.error) if outcome.error else st.rerun()
            if actions.button(t("delete", lang), key="del_" + doc.document_id):
                delete_document(doc.document_id)
                st.rerun()


# ===========================================================================
def ask(lang: str) -> None:
    from app.agents.pipeline import ask as run_pipeline

    st.title(t("ask_title", lang))

    if get_registry().stats()["documents"] == 0:
        st.info(t("empty_index", lang))
        return

    question = st.text_input(
        t("ask_a_question", lang),
        value=st.session_state.get("prefill", ""),
        placeholder=t("ask_placeholder", lang),
    )
    if st.button(t("ask_button", lang), type="primary"):
        if question.strip():
            st.session_state.prefill = ""
            with st.spinner(t("thinking", lang)):
                st.session_state.last_answer = run_pipeline(question, language=lang)

    st.caption(t("example_questions", lang))
    columns = st.columns(len(EXAMPLES.get(lang, EXAMPLES["en"])))
    for column, example in zip(columns, EXAMPLES.get(lang, EXAMPLES["en"])):
        if column.button(example, key="ex_" + example[:20], use_container_width=True):
            st.session_state.prefill = example
            st.rerun()

    result = st.session_state.get("last_answer")
    if result:
        st.divider()
        render_answer(result, lang)


def render_answer(result, lang: str) -> None:
    """Answer → sources → visual → semantic PCA."""
    from app.analytics.plots import semantic_map

    if result.error:
        st.error(t("gemini_failed", lang) + ": " + result.error)

    st.subheader(t("answer", lang))
    st.write(result.answer or "-")

    if result.key_findings:
        st.markdown("**" + t("key_findings", lang) + "**")
        for finding in result.key_findings:
            st.markdown("- " + finding)

    sources = result.sources
    if sources:
        st.subheader(t("sources", lang))
        for i, source in enumerate(sources, 1):
            st.markdown(
                "<div class='isa-src'>" + str(i) + ". " + source + "</div>",
                unsafe_allow_html=True,
            )

    # --- the visual, shown BEFORE its explanation ---
    visual = result.visual
    if visual and visual.found:
        st.divider()
        st.subheader(t("relevant_visual", lang))
        left, right = st.columns([1, 1])
        if Path(visual.image_path).exists():
            left.image(visual.image_path, use_container_width=True)
            left.markdown(
                "<div class='isa-src'>" + t("source", lang) + ": " + visual.citation
                + "</div>", unsafe_allow_html=True,
            )
        with right:
            st.markdown("**" + t("visual_analysis", lang) + "**")
            if visual.analysis:
                st.write(visual.analysis)
            elif visual.message:
                st.warning(visual.message)
    elif visual and visual.message:
        st.info(t("no_visual_found", lang))

    # --- semantic PCA ---
    st.divider()
    st.subheader(t("semantic_pca_map", lang))
    pca = result.pca
    if not pca or not pca.available:
        st.info(t("pca_unavailable", lang))
        return

    st.plotly_chart(semantic_map(pca, lang), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            "**" + t("query_cluster", lang) + ":** "
            + "<span class='isa-pill'>" + t("cluster", lang) + " "
            + str(pca.query_cluster) + "</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "**" + t("closest_cluster", lang) + ":** "
            + "<span class='isa-pill'>" + t("cluster", lang) + " "
            + str(pca.closest_cluster) + "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("**" + t("closest_to_question", lang) + ":**")
        for i, point in enumerate(pca.closest, 1):
            st.markdown(
                "<div class='isa-src'>" + str(i) + ". " + point.document + " — "
                + t("page", lang) + " " + str(point.page) + " — "
                + ("%.2f" % point.similarity) + "</div>",
                unsafe_allow_html=True,
            )
        if pca.nearby_other_cluster:
            st.markdown("**" + t("other_nearby", lang) + ":**")
            for point in pca.nearby_other_cluster:
                st.markdown(
                    "<div class='isa-src'>" + point.document + " — " + t("page", lang)
                    + " " + str(point.page) + " (" + t("cluster", lang) + " "
                    + str(point.cluster) + ")</div>",
                    unsafe_allow_html=True,
                )
    with right:
        st.markdown("**" + t("why_close", lang) + "**")
        st.markdown(
            "<div class='isa-quote'>" + (pca.explanation or "-") + "</div>",
            unsafe_allow_html=True,
        )

    st.caption(t("pca_limitation", lang))


# ===========================================================================
def data_analysis(lang: str) -> None:
    from app.analytics import plots
    from app.analytics.data_detector import (
        available_datasets, load_dataset, observations, profile_dataframe,
    )
    from app.analytics.pca import describe as describe_pca
    from app.analytics.pca import run_pca, top_loadings
    from app.document_processing.document_loader import save_dataset

    st.title(t("data_title", lang))

    upload = st.file_uploader(t("upload_dataset", lang), type=["csv", "xlsx", "xls"])
    if upload is not None:
        try:
            path = save_dataset(upload.getvalue(), upload.name)
            get_registry().add_dataset(path.name, str(path))
            st.rerun()
        except UnsafeFileError as exc:
            st.error(str(exc))

    datasets = available_datasets()
    if not datasets:
        st.info(t("no_dataset", lang))
        demo = Path(settings.upload_dir).parent.parent / "demo" / "model_benchmarks.csv"
        if demo.exists() and st.button(t("load_demo", lang)):
            path = save_dataset(demo.read_bytes(), demo.name)
            get_registry().add_dataset(path.name, str(path))
            st.rerun()
        return

    name = st.selectbox(t("dataset", lang), sorted(datasets))
    try:
        df = load_dataset(name)
    except Exception as exc:
        st.error(str(exc))
        return

    profile = profile_dataframe(df, name=name)
    columns = st.columns(4)
    columns[0].metric(t("rows", lang), profile.rows)
    columns[1].metric(t("columns", lang), profile.columns)
    columns[2].metric(t("numeric", lang), len(profile.numeric_features))
    columns[3].metric(t("missing", lang), sum(profile.missing_by_column.values()))

    with st.expander(t("preview", lang)):
        st.dataframe(df.head(25), use_container_width=True)
        for note in observations(profile):
            st.markdown("- " + note)

    if not profile.numeric_features:
        st.warning(t("no_numeric_columns", lang))
        return

    pca_tab, charts_tab = st.tabs(["PCA", t("charts", lang)])

    with pca_tab:
        features = st.multiselect(
            t("features", lang), profile.numeric_features,
            default=profile.numeric_features,
        )
        if st.button(t("run_pca", lang), type="primary"):
            result = run_pca(df, features=features, dataset_name=name)
            if not result.performed:
                st.error(result.reason)
            else:
                metrics = st.columns(min(result.components, 4))
                for i, ratio in enumerate(result.explained_variance[:4]):
                    metrics[i].metric("PC" + str(i + 1), "%.1f%%" % (ratio * 100))
                st.plotly_chart(plots.dataset_pca_scatter(result, lang=lang),
                                use_container_width=True)
                st.plotly_chart(plots.scree(result, lang=lang), use_container_width=True)
                for line in describe_pca(result, lang):
                    st.markdown("- " + line)
                for component in range(min(result.components, 2)):
                    st.markdown(
                        "**PC" + str(component + 1) + "**: "
                        + ", ".join(
                            d["feature"] + " (" + str(d["loading"]) + ")"
                            for d in top_loadings(result, component, 5)
                        )
                    )

    with charts_tab:
        numeric = profile.numeric_features
        categorical = profile.categorical_features
        kind = st.selectbox(t("charts", lang),
                            ["correlation", "histogram", "scatter", "bar"])
        try:
            if kind == "correlation":
                figure = plots.correlation_heatmap(df, lang)
            elif kind == "histogram":
                figure = plots.histogram(df, st.selectbox(t("columns", lang), numeric), lang)
            elif kind == "scatter":
                a, b, c = st.columns(3)
                x = a.selectbox("X", numeric, key="sx")
                y = b.selectbox("Y", numeric, index=min(1, len(numeric) - 1), key="sy")
                colour = c.selectbox("Color", ["-"] + categorical, key="sc")
                figure = plots.scatter(df, x, y, None if colour == "-" else colour, lang)
            else:
                a, b = st.columns(2)
                category = a.selectbox(t("columns", lang),
                                       categorical or profile.column_names, key="bc")
                value = b.selectbox(t("numeric", lang), ["-"] + numeric, key="bv")
                figure = plots.bar(df, category, None if value == "-" else value, lang)
            st.plotly_chart(figure, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


# ===========================================================================
def reports(lang: str) -> None:
    from app.reports.generator import list_reports, render, to_markdown

    st.title(t("reports_title", lang))
    result = st.session_state.get("last_answer")

    if result is None:
        st.info(t("no_report_source", lang))
    else:
        st.markdown("**" + t("question", lang) + ":** " + result.query)
        title = st.text_input(t("report_title_label", lang), value=result.query[:90])
        formats = st.multiselect(t("formats", lang), ["pdf", "html", "markdown"],
                                 default=["pdf", "html", "markdown"])

        if st.button(t("generate_report", lang), type="primary", disabled=not formats):
            with st.spinner(t("thinking", lang)):
                result.query = title or result.query
                files = render(result, formats, lang)
            if files.get("pdf_error"):
                st.warning("PDF: " + files["pdf_error"])
            written = {k: v for k, v in files.items() if "error" not in k}
            if written:
                get_registry().bump("reports")
                columns = st.columns(len(written))
                mimes = {"markdown": "text/markdown", "html": "text/html",
                         "pdf": "application/pdf"}
                for column, (kind, path) in zip(columns, written.items()):
                    column.download_button(
                        t("download", lang) + " " + kind.upper(),
                        data=Path(path).read_bytes(), file_name=Path(path).name,
                        mime=mimes.get(kind, "application/octet-stream"),
                        use_container_width=True,
                    )
                with st.expander(t("preview", lang)):
                    st.markdown(to_markdown(result, lang))

    st.divider()
    st.subheader(t("previous_reports", lang))
    files = list_reports()
    if not files:
        st.caption("—")
        return
    for path in files[:15]:
        row = st.columns([4, 1])
        row[0].markdown("**" + path.name + "**")
        row[0].markdown(
            "<span class='isa-src'>" + human_size(path.stat().st_size) + "</span>",
            unsafe_allow_html=True,
        )
        row[1].download_button(t("download", lang), data=path.read_bytes(),
                               file_name=path.name, key="dl_" + path.name,
                               use_container_width=True)


# ===========================================================================
def settings_page(lang: str) -> None:
    import os

    from app.rag.embeddings import get_embedding_provider
    from app.rag.vector_store import get_vector_store

    st.title(t("settings_title", lang))

    model = st.text_input(t("gemini_model", lang), value=settings.gemini_model)
    left, right = st.columns(2)
    top_k = left.slider(t("top_k", lang), 1, 15, settings.top_k)
    max_points = right.slider(t("pca_max_points", lang), 20, 300, settings.pca_max_points, 10)

    if st.button(t("apply", lang), type="primary"):
        os.environ["GEMINI_MODEL"] = model
        os.environ["TOP_K"] = str(top_k)
        os.environ["PCA_MAX_POINTS"] = str(max_points)
        reload_settings()
        from app.models.gemini import reset_gemini

        reset_gemini()
        st.success(t("applied", lang))
        st.rerun()

    st.divider()
    st.subheader(t("current_config", lang))
    st.json(settings.redacted())

    st.markdown(
        "<div class='isa-src'>Embeddings: " + get_embedding_provider().name
        + " · Vectors: " + str(get_vector_store().count()) + "</div>",
        unsafe_allow_html=True,
    )
    if not settings.gemini_configured:
        st.warning(t("no_api_key", lang), icon="⚠️")
