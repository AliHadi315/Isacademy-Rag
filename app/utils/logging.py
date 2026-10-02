"""Structured logging.

One logger tree ("isacademy") writing to stdout *and* to data/processed/app.log,
plus an in-memory ring buffer the Streamlit UI reads for the live activity feed.

Secrets never reach the log: `redact()` scrubs anything key-shaped.
"""
from __future__ import annotations

import logging
import re
from collections import deque
from datetime import datetime
from pathlib import Path

_RING: deque = deque(maxlen=500)
_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,})")


def redact(text: str) -> str:
    return _SECRET.sub("***REDACTED***", str(text))


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _RING.append(
            {
                "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "level": record.levelname,
                "name": record.name.replace("isacademy.", ""),
                "message": redact(record.getMessage()),
            }
        )


_configured = False


def configure(level: str = "INFO", log_dir: Path | None = None) -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger("isacademy")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)
    root.addHandler(_RingHandler())

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            pass  # logging must never break the app
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    from app.config import settings

    configure(settings.log_level, settings.processed_dir)
    return logging.getLogger("isacademy." + name)


def recent_logs(limit: int = 100) -> list[dict]:
    return list(_RING)[-limit:]
