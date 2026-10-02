"""Semantic PCA, clustering, similarity, dataset PCA and the visual path."""
import pandas as pd
import pytest

from app.analytics.clustering import assign_cluster, choose_k, cluster_terms, cluster_vectors
from app.analytics.pca import describe, run_pca, top_loadings
from app.analytics.semantic_pca import build_semantic_pca
from app.analytics.similarity import cosine_to_query
from app.models.schemas import Chunk, ContentType, QueryPlan
from app.rag.embeddings import get_embedding_provider

CORPUS = [
    ("Principal component analysis retained 89 percent of the variance in two components.", 3),
    ("The explained variance of the first component was the largest by far.", 4),
    ("Validation accuracy improved from 71.2 to 88.4 percent after fine-tuning.", 5),
    ("Adaptive fine-tuning converged faster than the frozen baseline.", 6),
    ("Serving costs fell 28 percent quarter over quarter after the migration.", 9),
    ("Cold starts remain the dominant tail-latency risk in production.", 10),
]


def _index(store):
    provider = get_embedding_provider()
    chunks = [
        Chunk(chunk_id="c" + str(i), document_id="d1", document_name="paper.pdf",
              page_number=page, content=text, content_type=ContentType.TEXT,
              char_count=len(text), chunk_index=i)
        for i, (text, page) in enumerate(CORPUS)
    ]
    store.add(chunks, provider.embed_documents([c.content for c in chunks]))


# ------------------------------------------------------------- similarity
def test_cosine_is_computed_on_real_vectors():
    provider = get_embedding_provider()
    query = provider.embed_query("explained variance")
    docs = provider.embed_documents([
        "The explained variance of the first component was largest.",
        "Cold starts remain the dominant latency risk.",
    ])
    scores = cosine_to_query(query, docs)
    assert len(scores) == 2
    assert scores[0] > scores[1]
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_cosine_of_empty_documents():
    assert cosine_to_query([0.1, 0.2], []) == []


# ------------------------------------------------------------- clustering
def test_choose_k_respects_data_size():
    assert choose_k(2, 3) == 1
    assert choose_k(30, 3) == 3
    assert choose_k(7, 5) == 2


def test_cluster_vectors_and_assign():
    provider = get_embedding_provider()
    vectors = provider.embed_documents([text for text, _ in CORPUS])
    labels, model = cluster_vectors(vectors, 2)
    assert len(labels) == len(CORPUS)
    assert set(labels) <= {0, 1}
    assigned = assign_cluster(model, provider.embed_query("explained variance"))
    assert assigned in set(labels)


def test_cluster_terms_are_real_words():
    labels = [0, 0, 1, 1]
    terms = cluster_terms(
        ["variance components analysis", "variance explained analysis",
         "latency cold starts", "latency production risk"],
        labels,
    )
    assert set(terms) == {0, 1}
    assert all(isinstance(w, str) for words in terms.values() for w in words)


# ----------------------------------------------------------- semantic PCA
def test_semantic_pca_projects_query_and_documents(clean_index):
    _index(clean_index)
    provider = get_embedding_provider()
    vector = provider.embed_query("what was the explained variance?")
    pca = build_semantic_pca("what was the explained variance?", vector, explain=False)

    assert pca.available, pca.reason
    assert len(pca.points) == len(CORPUS) + 1          # documents + the query
    query_points = [p for p in pca.points if p.is_query]
    assert len(query_points) == 1
    assert len(pca.explained_variance) == 2


def test_semantic_pca_ranks_the_right_passage_first(clean_index):
    _index(clean_index)
    provider = get_embedding_provider()
    vector = provider.embed_query("principal component analysis explained variance")
    pca = build_semantic_pca("principal component analysis explained variance",
                             vector, explain=False)
    assert pca.closest
    assert "variance" in pca.closest[0].snippet.lower()
    assert pca.closest[0].similarity >= pca.closest[-1].similarity


def test_semantic_pca_reports_clusters(clean_index):
    _index(clean_index)
    provider = get_embedding_provider()
    pca = build_semantic_pca("cold start latency",
                             provider.embed_query("cold start latency"), explain=False)
    assert pca.query_cluster >= 0
    assert pca.closest_cluster >= 0
    assert pca.n_clusters >= 1
    assert all(p.cluster >= 0 for p in pca.points if not p.is_query)


def test_semantic_pca_similarity_is_not_from_the_projection(clean_index):
    """Scores must come from the full vectors, so they stay in [-1, 1]."""
    _index(clean_index)
    provider = get_embedding_provider()
    pca = build_semantic_pca("variance", provider.embed_query("variance"), explain=False)
    assert all(-1.0 <= p.similarity <= 1.0 for p in pca.points)


def test_semantic_pca_declines_on_an_empty_index(clean_index):
    provider = get_embedding_provider()
    pca = build_semantic_pca("anything", provider.embed_query("anything"), explain=False)
    assert not pca.available and pca.reason


def test_semantic_pca_declines_with_too_few_points(clean_index):
    provider = get_embedding_provider()
    chunk = Chunk(chunk_id="only", document_id="d", document_name="a.pdf",
                  page_number=1, content="A single lonely passage of text.",
                  char_count=30, chunk_index=0)
    clean_index.add([chunk], provider.embed_documents([chunk.content]))
    pca = build_semantic_pca("text", provider.embed_query("text"), explain=False)
    assert not pca.available
    assert pca.reason == "not_enough_points"


def test_explanation_falls_back_without_an_api_key(clean_index):
    """No key must not mean an empty panel - the fallback names real terms."""
    _index(clean_index)
    provider = get_embedding_provider()
    pca = build_semantic_pca("explained variance",
                             provider.embed_query("explained variance"), explain=True)
    assert pca.available
    assert pca.explanation          # never blank


# ----------------------------------------------------------- dataset PCA
def test_dataset_pca_runs(sample_dataframe):
    result = run_pca(sample_dataframe, dataset_name="test")
    assert result.performed
    assert result.components == 2
    assert 0.0 < sum(result.explained_variance) <= 1.0 + 1e-9
    assert len(result.coordinates) == len(sample_dataframe)
    assert len(result.clusters) == len(sample_dataframe)


def test_dataset_pca_excludes_constant_columns(sample_dataframe):
    result = run_pca(sample_dataframe, dataset_name="test")
    assert "constant" not in result.features


def test_correlated_features_drive_pc1(sample_dataframe):
    result = run_pca(sample_dataframe, dataset_name="test")
    drivers = {d["feature"] for d in top_loadings(result, 0, 2)}
    assert {"feature_a", "feature_b"} & drivers


@pytest.mark.parametrize("df", [
    pd.DataFrame({"only": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}),      # one feature
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 5.0]}),            # too few rows
    pd.DataFrame({"a": [1.0] * 10, "b": [2.0] * 10}),            # all constant
    pd.DataFrame({"txt": list("abcdefghij")}),                   # not numeric
])
def test_dataset_pca_refuses_unusable_data(df):
    result = run_pca(df)
    assert not result.performed and result.reason


def test_describe_is_bilingual(sample_dataframe):
    result = run_pca(sample_dataframe)
    assert any("variance" in line for line in describe(result, "en"))
    assert any("التباين" in line for line in describe(result, "ar"))


# ---------------------------------------------------------------- visual
def test_page_render_produces_an_image(sample_pdf):
    from app.document_processing.pdf_extractor import render_page_image

    path = render_page_image(sample_pdf, 1)
    assert path.exists() and path.stat().st_size > 1000


def test_page_render_rejects_a_missing_page(sample_pdf):
    from app.document_processing.pdf_extractor import PDFExtractionError, render_page_image

    with pytest.raises(PDFExtractionError):
        render_page_image(sample_pdf, 999)


def test_find_visual_prefers_an_extracted_figure(sample_pdf, clean_index):
    from app.document_processing.processor import index_path
    from app.vision.gemini_vision import find_visual

    index_path(sample_pdf)
    plan = QueryPlan(original_query="What does Figure 1 show?", figure_hint="1",
                     needs_visual=True)
    visual = find_visual("What does Figure 1 show?", plan)
    assert visual.found
    from pathlib import Path

    assert Path(visual.image_path).exists()
    assert visual.page >= 1


def test_find_visual_falls_back_to_a_page_render(sample_pdf, clean_index):
    """A visual question must never come back with nothing to show."""
    from app.agents.retrieval_agent import RetrievalAgent
    from app.document_processing.processor import index_path
    from app.vision.gemini_vision import find_visual

    index_path(sample_pdf)
    plan = QueryPlan(original_query="Explain the chart on page 1.", page_hint=1,
                     needs_visual=True)
    retrieval = RetrievalAgent().run("chart")
    visual = find_visual("Explain the chart on page 1.", plan, retrieval)
    assert visual.found
    assert visual.page == 1


def test_visual_analysis_reports_a_missing_key(sample_pdf, clean_index):
    from app.document_processing.processor import index_path
    from app.vision.gemini_vision import analyze_visual, find_visual

    index_path(sample_pdf)
    plan = QueryPlan(original_query="What does Figure 1 show?", figure_hint="1")
    visual = analyze_visual(find_visual("Figure 1", plan), "Figure 1", "", "en")
    # no API key in tests: it must report that, not fabricate an analysis
    assert not visual.analysis
    assert "api key" in visual.message.lower()
