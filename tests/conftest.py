"""Test fixtures - a throwaway data directory, no API key, no downloads."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="isacademy_tests_"))
os.environ.update({
    "CHROMA_PATH": str(_TMP / "vector_db"),
    "UPLOAD_DIR": str(_TMP / "uploads"),
    "OUTPUT_DIR": str(_TMP / "reports"),
    "PROCESSED_DIR": str(_TMP / "processed"),
    "IMAGES_DIR": str(_TMP / "images"),
    "DATASETS_DIR": str(_TMP / "datasets"),
    "VIZ_DIR": str(_TMP / "viz"),
    "EMBEDDING_MODEL": "hashing",     # deterministic, no download
    "GEMINI_API_KEY": "",             # tests never call the real API
    "LOG_LEVEL": "WARNING",
})

import pytest  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def sample_pdf(tmp_root) -> Path:
    """A real PDF with text, an embedded chart and a captioned figure."""
    pytest.importorskip("reportlab")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table

    chart = tmp_root / "chart.png"
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot([1, 2, 3, 4], [10, 30, 45, 70], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(chart, dpi=110)
    plt.close(fig)

    pdf = tmp_root / "sample_study.pdf"
    styles = getSampleStyleSheet()
    SimpleDocTemplate(str(pdf), pagesize=A4, title="Sample Study").build([
        Paragraph("Sample Study on Retrieval Quality", styles["Title"]),
        Paragraph(
            "Retrieval accuracy improved from 61.4 percent to 84.9 percent after "
            "reranking was enabled. Latency increased by 12 milliseconds.",
            styles["BodyText"],
        ),
        Spacer(1, 0.4 * cm),
        Image(str(chart), width=10 * cm, height=6.2 * cm),
        Paragraph("Figure 1: Accuracy per epoch during reranking.", styles["Italic"]),
        Spacer(1, 0.4 * cm),
        Table([["System", "Accuracy"], ["baseline", "61.4"], ["reranked", "84.9"]]),
    ])
    return pdf


@pytest.fixture(scope="session")
def sample_dataframe():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 60
    a = rng.normal(50, 12, n)
    return pd.DataFrame({
        "group": rng.choice(["x", "y"], n),
        "feature_a": a.round(3),
        "feature_b": (a * 1.8 + rng.normal(0, 3, n)).round(3),
        "feature_c": rng.normal(5, 2, n).round(3),
        "constant": 7.0,
    })


@pytest.fixture
def clean_index():
    from app.document_processing.metadata import get_registry
    from app.rag.vector_store import get_vector_store

    store = get_vector_store()
    store.reset()
    registry = get_registry()
    for doc in registry.all():
        registry.remove(doc.document_id)
    yield store
    store.reset()
