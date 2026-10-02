"""Typed data contracts shared by every module.

Flow:
    QueryPlan -> RetrievalResult -> (VisualAnswer | SemanticPCAResult)
              -> AnswerResult -> report
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
class Intent(str, Enum):
    RAG = "RAG"                       # plain text question
    VISUAL = "VISUAL"                 # about a chart / figure / table image
    DATA = "DATA"                     # about numbers in a dataset
    REPORT = "REPORT"                 # asks for a report
    MIXED = "MIXED"                   # visual + text


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FIGURE = "figure"


class ExtractedImage(BaseModel):
    image_id: str
    document_name: str
    page_number: int
    path: str
    width: int = 0
    height: int = 0
    caption: str = ""
    is_page_render: bool = False      # a whole-page fallback image, not a figure


class ExtractedTable(BaseModel):
    table_id: str
    document_name: str
    page_number: int
    rows: list[list[str]] = Field(default_factory=list)
    markdown: str = ""

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)


class DocumentPage(BaseModel):
    document_name: str
    page_number: int
    text: str = ""
    images: list[ExtractedImage] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    ocr_used: bool = False


class Document(BaseModel):
    document_id: str                  # sha256 of the file bytes
    name: str
    path: str
    file_type: str
    size_bytes: int = 0
    page_count: int = 0
    chunk_count: int = 0
    image_count: int = 0
    table_count: int = 0
    indexed_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    title: str = ""
    author: str = ""
    ocr_pages: int = 0


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    content: str
    content_type: ContentType = ContentType.TEXT
    image_path: str = ""              # set for figure/table chunks
    char_count: int = 0
    chunk_index: int = 0

    def metadata(self) -> dict:
        """Flat, primitive-only - what Chroma accepts."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "content_type": self.content_type.value,
            "image_path": self.image_path,
            "chunk_index": self.chunk_index,
        }


# --------------------------------------------------------------------------
# Query / retrieval
# --------------------------------------------------------------------------
class QueryPlan(BaseModel):
    original_query: str
    intent: Intent = Intent.RAG
    search_query: str = ""
    needs_text: bool = True
    needs_visual: bool = False
    needs_data: bool = False
    wants_report: bool = False
    figure_hint: str = ""             # e.g. "3" from "what does Figure 3 show?"
    page_hint: int = 0                # e.g. 15 from "the graph on page 15"
    reason: str = ""


class SearchResult(BaseModel):
    content: str
    document: str
    document_id: str = ""
    page: int = 0
    chunk_id: str = ""
    score: float = 0.0
    content_type: str = "text"
    image_path: str = ""

    @property
    def citation(self) -> str:
        return self.document + " — Page " + str(self.page)


class RetrievalResult(BaseModel):
    query: str = ""
    results: list[SearchResult] = Field(default_factory=list)
    threshold: float = 0.0
    sufficient: bool = False
    message: str = ""


# --------------------------------------------------------------------------
# Visual
# --------------------------------------------------------------------------
class VisualAnswer(BaseModel):
    found: bool = False
    image_path: str = ""
    document: str = ""
    page: int = 0
    caption: str = ""
    is_page_render: bool = False
    analysis: str = ""
    message: str = ""

    @property
    def citation(self) -> str:
        label = self.caption or ("Page " + str(self.page))
        return self.document + " — Page " + str(self.page) + (" — " + label if self.caption else "")


# --------------------------------------------------------------------------
# Semantic PCA
# --------------------------------------------------------------------------
class PCAPoint(BaseModel):
    x: float
    y: float
    label: str                        # "document.pdf p.12" or "Your question"
    document: str = ""
    page: int = 0
    cluster: int = -1
    similarity: float = 0.0
    is_query: bool = False
    snippet: str = ""


class SemanticPCAResult(BaseModel):
    available: bool = False
    reason: str = ""
    points: list[PCAPoint] = Field(default_factory=list)
    query_cluster: int = -1
    closest_cluster: int = -1        # dominant cluster among the closest items
    n_clusters: int = 0
    explained_variance: list[float] = Field(default_factory=list)
    closest: list[PCAPoint] = Field(default_factory=list)
    nearby_other_cluster: list[PCAPoint] = Field(default_factory=list)
    explanation: str = ""             # "why are they close?" - written by Gemini
    cluster_terms: dict[int, list[str]] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Dataset analysis
# --------------------------------------------------------------------------
class PCAResult(BaseModel):
    performed: bool = False
    reason: str = ""
    components: int = 0
    features: list[str] = Field(default_factory=list)
    explained_variance: list[float] = Field(default_factory=list)
    cumulative_variance: list[float] = Field(default_factory=list)
    loadings: dict[str, list[float]] = Field(default_factory=dict)
    coordinates: list[list[float]] = Field(default_factory=list)
    clusters: list[int] = Field(default_factory=list)
    n_samples: int = 0


class DatasetProfile(BaseModel):
    name: str
    path: str = ""
    rows: int = 0
    columns: int = 0
    column_names: list[str] = Field(default_factory=list)
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    missing_by_column: dict[str, int] = Field(default_factory=dict)
    describe: dict[str, dict[str, float]] = Field(default_factory=dict)
    top_correlations: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Final answer
# --------------------------------------------------------------------------
class AnswerResult(BaseModel):
    query: str
    language: str = "en"
    answer: str = ""
    key_findings: list[str] = Field(default_factory=list)
    plan: Optional[QueryPlan] = None
    retrieval: Optional[RetrievalResult] = None
    visual: Optional[VisualAnswer] = None
    pca: Optional[SemanticPCAResult] = None
    grounded: bool = True             # False when the system had to decline
    error: str = ""
    duration_ms: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def sources(self) -> list[str]:
        seen, out = set(), []
        for r in (self.retrieval.results if self.retrieval else []):
            key = (r.document, r.page)
            if key not in seen:
                seen.add(key)
                out.append(r.citation)
        if self.visual and self.visual.found:
            key = (self.visual.document, self.visual.page)
            if key not in seen:
                out.append(self.visual.citation)
        return out
