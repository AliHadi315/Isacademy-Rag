"""Launcher.

    python run.py           start the app
    python run.py --demo    build + index the demo documents, then start
    python run.py --check   dependency / configuration check
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

APP = ROOT / "app" / "ui" / "streamlit_app.py"


def check() -> int:
    required = {
        "streamlit": "UI",
        "plotly": "interactive charts",
        "google.genai": "Gemini (text + vision)",
        "chromadb": "vector database",
        "pymupdf": "PDF extraction",
        "sklearn": "PCA and clustering",
        "pandas": "dataframes",
        "pydantic": "schemas",
    }
    optional = {
        "sentence_transformers": "semantic embeddings (falls back to hashing)",
        "reportlab": "PDF report export",
        "dotenv": ".env loading",
        "easyocr": "OCR for scanned PDFs",
    }

    missing = 0
    print("Required:")
    for module, why in required.items():
        try:
            __import__(module)
            print("  OK      " + module + " - " + why)
        except ImportError:
            missing += 1
            print("  MISSING " + module + " - " + why)

    print("\nOptional:")
    for module, why in optional.items():
        try:
            __import__(module)
            print("  OK      " + module + " - " + why)
        except ImportError:
            print("  absent  " + module + " - " + why)

    from app.config import settings

    print("\nConfiguration:")
    for key, value in settings.redacted().items():
        print("  " + key + " = " + str(value))
    if not settings.gemini_configured:
        print("\n  ! GEMINI_API_KEY is not set - add it to .env before asking questions.")
    if missing:
        print("\nInstall the missing packages: pip install -r requirements.txt")
    return 1 if missing else 0


def build_demo() -> None:
    from demo.make_demo_data import main as make_demo

    print("Building demo data...")
    make_demo()

    from app.document_processing.document_loader import save_dataset
    from app.document_processing.metadata import get_registry
    from app.document_processing.processor import index_path

    for pdf in sorted((ROOT / "demo").glob("*.pdf")):
        outcome = index_path(pdf)
        status = "already indexed" if outcome.skipped else (
            outcome.error or str(outcome.document.chunk_count) + " chunks"
        )
        print("  " + pdf.name + ": " + status)

    for csv in sorted((ROOT / "demo").glob("*.csv")):
        path = save_dataset(csv.read_bytes(), csv.name)
        get_registry().add_dataset(path.name, str(path))
        print("  " + csv.name + ": registered")


def serve() -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed. Run: pip install -r requirements.txt")
        return 1
    return subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.headless", "true",
    ])


def main() -> int:
    args = set(sys.argv[1:])
    if "--check" in args:
        return check()
    if args & {"--demo", "--demo-only"}:
        build_demo()
        if "--demo-only" in args:
            return 0
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
