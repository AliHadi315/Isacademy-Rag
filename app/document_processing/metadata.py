"""Document registry - the persistent catalogue of what has been indexed.

WHAT   A tiny JSON-backed store (`data/processed/registry.json`) keyed by the
       SHA-256 of the file bytes, holding one `Document` record per file plus
       the extracted images and dataset paths belonging to it.
WHY    Two jobs: (1) survive restarts so the Dashboard is not empty every run,
       and (2) content-hash dedup - the same PDF uploaded twice is never
       re-embedded (spec 46).
IN     Document / ExtractedImage records
OUT    Document records, image records, counters
NEXT   -> Dashboard, Documents page, vision_agent (image lookup)
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app.config import settings
from app.models.schemas import Document, ExtractedImage
from app.utils.logging import get_logger

log = get_logger("registry")
_LOCK = threading.Lock()   # ponytail: one global lock; per-document locks only if concurrent uploads become real


class DocumentRegistry:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.processed_dir / "registry.json")
        self._data = {"documents": {}, "images": {}, "datasets": {}, "counters": {}}
        self._load()

    # -- persistence -------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._data.update(json.loads(self.path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("registry unreadable (%s) - starting fresh", exc)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            log.error("could not persist registry: %s", exc)

    # -- documents ---------------------------------------------------
    def has(self, document_id: str) -> bool:
        return document_id in self._data["documents"]

    def get(self, document_id: str) -> Document | None:
        raw = self._data["documents"].get(document_id)
        return Document(**raw) if raw else None

    def by_name(self, name: str) -> Document | None:
        for raw in self._data["documents"].values():
            if raw.get("name") == name:
                return Document(**raw)
        return None

    def add(self, doc: Document) -> None:
        with _LOCK:
            self._data["documents"][doc.document_id] = doc.model_dump()
            self._save()
        log.info("registry: indexed %s (%d chunks)", doc.name, doc.chunk_count)

    def remove(self, document_id: str) -> None:
        with _LOCK:
            self._data["documents"].pop(document_id, None)
            self._data["images"].pop(document_id, None)
            self._save()

    def all(self) -> list[Document]:
        return [Document(**d) for d in self._data["documents"].values()]

    # -- images ------------------------------------------------------
    def add_images(self, document_id: str, images: list[ExtractedImage]) -> None:
        with _LOCK:
            self._data["images"][document_id] = [i.model_dump() for i in images]
            self._save()

    def images(self, document_id: str | None = None) -> list[ExtractedImage]:
        buckets = (
            [self._data["images"].get(document_id, [])]
            if document_id
            else list(self._data["images"].values())
        )
        return [ExtractedImage(**i) for bucket in buckets for i in bucket]

    # -- datasets ----------------------------------------------------
    def add_dataset(self, name: str, path: str, source: str = "uploaded") -> None:
        with _LOCK:
            self._data["datasets"][name] = {"path": path, "source": source}
            self._save()

    def datasets(self) -> dict:
        return dict(self._data["datasets"])

    def remove_dataset(self, name: str) -> None:
        with _LOCK:
            self._data["datasets"].pop(name, None)
            self._save()

    # -- counters (dashboard) ---------------------------------------
    def bump(self, key: str, amount: int = 1) -> None:
        with _LOCK:
            self._data["counters"][key] = self._data["counters"].get(key, 0) + amount
            self._save()

    def counter(self, key: str) -> int:
        return int(self._data["counters"].get(key, 0))

    def stats(self) -> dict:
        docs = self.all()
        return {
            "documents": len(docs),
            "pages": sum(d.page_count for d in docs),
            "chunks": sum(d.chunk_count for d in docs),
            "images": sum(d.image_count for d in docs),
            "tables": sum(d.table_count for d in docs),
            "datasets": len(self._data["datasets"]),
            "questions": self.counter("questions"),
            "reports": self.counter("reports"),
        }


_registry: DocumentRegistry | None = None


def get_registry() -> DocumentRegistry:
    global _registry
    if _registry is None:
        _registry = DocumentRegistry()
    return _registry
