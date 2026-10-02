"""Small shared helpers used across agents, RAG and the UI."""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime


def clean_text(text: str) -> str:
    """Normalise unicode, undo hyphen line-wraps, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                 # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # de-hyphenate wrapped words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list:
    """Cheap sentence splitter - good enough for claim extraction / snippets."""
    flat = clean_text(text).replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])", flat)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def truncate(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def parse_json_block(raw: str):
    """Pull the first JSON object/array out of an LLM response.

    Models like to wrap JSON in prose or code fences; this survives both and
    returns None instead of raising so callers can fall back to heuristics.
    """
    if not raw:
        return None
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "was", "one", "our",
    "out", "has", "have", "this", "that", "with", "from", "they", "what", "which", "their",
    "there", "about", "would", "these", "does", "into", "than", "them", "some", "were", "when",
    "where", "will", "your", "how", "why", "who", "show", "give", "tell", "using", "used", "also",
    "its", "each", "such", "been", "over", "more", "most", "between", "based", "per", "may",
}


def keywords(text: str, limit: int = 12) -> list:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", (text or "").lower())
    counts: dict = {}
    for w in words:
        if w not in STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:limit]


def lexical_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words - used by the verification agent."""
    sa, sb = set(keywords(a, 60)), set(keywords(b, 200))
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
