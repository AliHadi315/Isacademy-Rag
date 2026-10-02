"""Embeddings, ChromaDB, retrieval, citations and ingestion."""
import pytest

from app.agents.retrieval_agent import RetrievalAgent
from app.document_processing.metadata import get_registry
from app.document_processing.processor import index_path, index_upload
from app.models.schemas import Chunk, ContentType
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.utils.file_utils import UnsafeFileError, safe_filename, sha256_bytes, validate_upload

CORPUS = [
    ("Principal component analysis reduced the feature space to two components "
     "retaining 89 percent of the variance.", 3),
    ("Validation accuracy improved from 71.2 percent to 88.4 percent after "
     "adaptive fine-tuning.", 5),
    ("Serving costs fell 28 percent quarter over quarter after the migration.", 9),
]


def _index(store, document_id="doc1"):
    provider = get_embedding_provider()
    chunks = [
        Chunk(chunk_id=document_id + "_c" + str(i), document_id=document_id,
              document_name="paper.pdf", page_number=page, content=text,
              content_type=ContentType.TEXT, char_count=len(text), chunk_index=i)
        for i, (text, page) in enumerate(CORPUS)
    ]
    store.add(chunks, provider.embed_documents([c.content for c in chunks]))
    return chunks


# ------------------------------------------------------------- embeddings
def test_provider_interface():
    provider = get_embedding_provider()
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension > 0


def test_vectors_are_unit_norm():
    import math

    for vector in get_embedding_provider().embed_documents(["hello world", "another"]):
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-4)


def test_relevant_scores_higher_than_irrelevant():
    provider = get_embedding_provider()
    query = provider.embed_query("principal component analysis explained variance")
    relevant, irrelevant = provider.embed_documents([
        "We ran principal component analysis and report the explained variance.",
        "The cat sat quietly on a warm windowsill.",
    ])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(query, relevant) > dot(query, irrelevant)


def test_empty_input():
    assert get_embedding_provider().embed_documents([]) == []


# ----------------------------------------------------------- vector store
def test_store_roundtrip(clean_index):
    _index(clean_index)
    assert clean_index.count() == len(CORPUS)


def test_neighborhood_returns_vectors(clean_index):
    _index(clean_index)
    vector = get_embedding_provider().embed_query("explained variance")
    results, vectors = clean_index.neighborhood(vector, 3)
    assert len(results) == len(vectors) == 3
    assert len(vectors[0]) == get_embedding_provider().dimension


def test_delete_document(clean_index):
    _index(clean_index, "docA")
    removed = clean_index.delete_document("docA")
    assert removed == len(CORPUS) and clean_index.count() == 0


def test_metadata_survives_the_roundtrip(clean_index):
    _index(clean_index)
    hits = RetrievalAgent().run("principal component analysis").results
    assert hits[0].document == "paper.pdf"
    assert hits[0].page in {3, 5, 9}          # a real page, never invented
    assert hits[0].chunk_id


# --------------------------------------------------------------- retrieval
def test_retrieves_the_right_passage(clean_index):
    _index(clean_index)
    result = RetrievalAgent().run("explained variance of the principal components")
    assert result.sufficient
    assert "principal component" in result.results[0].content.lower()
    assert result.results[0].page == 3


def test_scores_descend(clean_index):
    _index(clean_index)
    scores = [r.score for r in RetrievalAgent().run("accuracy after fine-tuning").results]
    assert scores == sorted(scores, reverse=True)


def test_threshold_can_reject_everything(clean_index):
    _index(clean_index)
    result = RetrievalAgent(threshold=0.99).run("accuracy")
    assert not result.sufficient and result.message


def test_empty_index_is_reported(clean_index):
    result = RetrievalAgent().run("anything")
    assert not result.sufficient
    assert "no documents" in result.message.lower()


def test_document_filter(clean_index):
    _index(clean_index, "docA")
    _index(clean_index, "docB")
    result = RetrievalAgent().run("principal component", document_ids=["docA"])
    assert result.results and all(r.document_id == "docA" for r in result.results)


def test_context_block_is_numbered(clean_index):
    _index(clean_index)
    block = RetrievalAgent.context_block(RetrievalAgent().run("accuracy").results)
    assert "[1]" in block and "page" in block


def test_fingerprint_blocks_a_model_swap(clean_index):
    from app.rag.vector_store import check_fingerprint, record_fingerprint

    _index(clean_index)
    provider = get_embedding_provider()
    record_fingerprint(provider)
    assert check_fingerprint(provider) == ""

    class Other:
        name = "another-encoder"
        dimension = 768

    assert "re-index" in check_fingerprint(Other()).lower()


# --------------------------------------------------------------- ingestion
def test_index_pdf_end_to_end(sample_pdf, clean_index):
    outcome = index_path(sample_pdf)
    assert outcome.ok, outcome.error
    assert outcome.document.page_count >= 1
    assert outcome.document.chunk_count > 0
    assert clean_index.count() == outcome.document.chunk_count


def test_identical_upload_is_deduplicated(sample_pdf, clean_index):
    data = sample_pdf.read_bytes()
    first = index_upload(data, "study.pdf")
    assert first.ok and not first.skipped
    assert index_upload(data, "copy.pdf").skipped


def test_document_id_is_the_content_hash(sample_pdf, clean_index):
    data = sample_pdf.read_bytes()
    assert index_upload(data, "study.pdf").document.document_id == sha256_bytes(data)


def test_bad_upload_returns_an_error(clean_index):
    outcome = index_upload(b"not a pdf", "broken.pdf")
    assert not outcome.ok and outcome.error


def test_registry_stats(sample_pdf, clean_index):
    index_path(sample_pdf)
    stats = get_registry().stats()
    assert stats["documents"] >= 1 and stats["chunks"] > 0


# ---------------------------------------------------------------- security
@pytest.mark.parametrize("raw", ["../../etc/passwd", "..\\..\\windows\\cmd.exe"])
def test_safe_filename_strips_traversal(raw):
    safe = safe_filename(raw)
    assert ".." not in safe and "/" not in safe and "\\" not in safe


def test_rejects_unknown_extension():
    from app.config import settings

    with pytest.raises(UnsafeFileError):
        validate_upload("payload.exe", 100, settings.allowed_doc_ext, 10)


def test_rejects_oversized_file():
    from app.config import settings

    with pytest.raises(UnsafeFileError):
        validate_upload("big.pdf", 50 * 1024 * 1024, settings.allowed_doc_ext, 10)


def test_api_key_is_never_exposed():
    from app.config import settings

    assert "gemini_api_key" in settings.redacted()
    assert settings.redacted()["gemini_api_key"] in ("not set",) or \
        "chars" in settings.redacted()["gemini_api_key"]
