"""Filesystem helpers: safe names, hashing, size checks, dedup.

Security (spec 45): uploads are validated by extension AND size, filenames are
sanitised to a whitelist, and the resolved destination is verified to still sit
inside the target directory (path-traversal protection).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._\- ]+")


class UnsafeFileError(ValueError):
    """Raised when an upload fails validation."""


def safe_filename(name: str) -> str:
    """`../../etc/passwd` -> `etc_passwd`; keeps a readable, harmless name."""
    name = Path(str(name)).name           # strip any directory component
    name = name.replace("..", "_")
    name = _SAFE.sub("_", name).strip() or "unnamed"
    return name[:120]


def resolve_inside(directory: Path, filename: str) -> Path:
    """Return `directory/safe(filename)` and prove it did not escape."""
    directory = Path(directory).resolve()
    target = (directory / safe_filename(filename)).resolve()
    if target.parent != directory and directory not in target.parents:
        raise UnsafeFileError("Path traversal blocked for " + repr(filename))
    return target


def validate_upload(filename: str, size_bytes: int, allowed_ext: tuple, max_mb: int) -> str:
    ext = Path(safe_filename(filename)).suffix.lower()
    if ext not in allowed_ext:
        raise UnsafeFileError(
            "Unsupported file type '" + (ext or "none") + "'. Allowed: " + ", ".join(allowed_ext)
        )
    if size_bytes <= 0:
        raise UnsafeFileError("File is empty.")
    if size_bytes > max_mb * 1024 * 1024:
        raise UnsafeFileError(
            "File is %.1f MB - the limit is %d MB." % (size_bytes / 1e6, max_mb)
        )
    return ext


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return ("%.0f %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024
    return "%.1f TB" % n
