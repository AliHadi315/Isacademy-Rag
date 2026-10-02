"""Central configuration - read once from the environment.

Gemini is the only AI provider. The API key is read from the environment and
is never hardcoded, logged or displayed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _p(env: str, default: str) -> Path:
    path = Path(os.getenv(env, default))
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _i(env: str, default: int) -> int:
    try:
        return int(os.getenv(env, default))
    except (TypeError, ValueError):
        return default


def _f(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, default))
    except (TypeError, ValueError):
        return default


def _b(env: str, default: bool = False) -> bool:
    return os.getenv(env, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Gemini (the only AI provider) ---
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    )
    gemini_timeout: int = field(default_factory=lambda: _i("GEMINI_TIMEOUT", 90))

    # --- embeddings (local; not an LLM provider) ---
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    embedding_batch_size: int = field(default_factory=lambda: _i("EMBEDDING_BATCH_SIZE", 32))
    embedding_load_timeout: int = field(default_factory=lambda: _i("EMBEDDING_LOAD_TIMEOUT", 300))

    # --- chunking / retrieval ---
    chunk_size: int = field(default_factory=lambda: _i("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _i("CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _i("TOP_K", 5))
    similarity_threshold: float = field(default_factory=lambda: _f("SIMILARITY_THRESHOLD", 0.30))

    # --- semantic PCA ---
    pca_neighbors: int = field(default_factory=lambda: _i("PCA_NEIGHBORS", 25))
    pca_max_points: int = field(default_factory=lambda: _i("PCA_MAX_POINTS", 100))
    pca_clusters: int = field(default_factory=lambda: _i("PCA_CLUSTERS", 3))

    # --- UI ---
    default_language: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LANGUAGE", "en").strip().lower()
    )

    # --- storage ---
    chroma_path: Path = field(default_factory=lambda: _p("CHROMA_PATH", "./data/vector_db"))
    upload_dir: Path = field(default_factory=lambda: _p("UPLOAD_DIR", "./data/uploads"))
    output_dir: Path = field(default_factory=lambda: _p("OUTPUT_DIR", "./data/reports"))
    processed_dir: Path = field(default_factory=lambda: _p("PROCESSED_DIR", "./data/processed"))
    images_dir: Path = field(default_factory=lambda: _p("IMAGES_DIR", "./data/images"))
    datasets_dir: Path = field(default_factory=lambda: _p("DATASETS_DIR", "./data/datasets"))
    viz_dir: Path = field(default_factory=lambda: _p("VIZ_DIR", "./data/visualizations"))

    # --- limits ---
    max_upload_mb: int = field(default_factory=lambda: _i("MAX_UPLOAD_MB", 50))
    enable_ocr: bool = field(default_factory=lambda: _b("ENABLE_OCR", False))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    allowed_doc_ext: tuple = (".pdf", ".txt", ".md")
    allowed_data_ext: tuple = (".csv", ".xlsx", ".xls")

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    def redacted(self) -> dict:
        """Safe to display - never exposes the key itself."""
        return {
            "gemini_model": self.gemini_model,
            "gemini_api_key": (
                "set (" + str(len(self.gemini_api_key)) + " chars)"
                if self.gemini_api_key else "not set"
            ),
            "embedding_model": self.embedding_model,
            "chroma_path": str(self.chroma_path),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "pca_neighbors": self.pca_neighbors,
            "pca_max_points": self.pca_max_points,
            "default_language": self.default_language,
        }


settings = Settings()


def reload_settings() -> Settings:
    global settings
    settings = Settings()
    return settings
