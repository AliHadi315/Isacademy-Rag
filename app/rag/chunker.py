"""Structure-aware chunking.

WHAT   Splits each page into overlapping, sentence-aligned chunks and emits a
       separate chunk per table (kept whole, as Markdown) and per figure
       caption.  Every chunk keeps document_name + page_number + chunk_id.
WHY    Naive fixed-width slicing cuts sentences and tables in half, which both
       hurts embedding quality and produces unquotable evidence.  Overlap keeps
       cross-boundary facts retrievable.
LIB    stdlib only - the splitting rule is cheap and deterministic.
IN     list[DocumentPage] + document_id
OUT    list[Chunk]
NEXT   -> embeddings -> vector_store
"""
from __future__ import annotations

from app.config import settings
from app.models.schemas import Chunk, ContentType, DocumentPage
from app.utils.helpers import clean_text, split_sentences
from app.utils.logging import get_logger

log = get_logger("chunker")

MIN_CHUNK_CHARS = 40      # below this a chunk carries no retrievable meaning


def _pack(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Greedy sentence packing with a sentence-aligned tail overlap."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        # A single sentence longer than the window: hard-split it.
        if len(sentence) > size:
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            for i in range(0, len(sentence), size - overlap if size > overlap else size):
                chunks.append(sentence[i : i + size])
            continue

        if length + len(sentence) + 1 > size and current:
            chunks.append(" ".join(current))
            # rebuild the overlap tail out of whole sentences
            tail, tail_len = [], 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len

        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if len(c.strip()) >= MIN_CHUNK_CHARS]


def chunk_pages(
    pages: list[DocumentPage],
    document_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    size = chunk_size or settings.chunk_size
    overlap = min(chunk_overlap if chunk_overlap is not None else settings.chunk_overlap, size // 2)

    chunks: list[Chunk] = []
    index = 0

    for page in pages:
        base = document_id[:12] + "_p" + str(page.page_number)

        # 1. prose
        text = clean_text(page.text)
        if text:
            for i, body in enumerate(_pack(split_sentences(text), size, overlap)):
                chunks.append(
                    Chunk(
                        chunk_id=base + "_c" + str(i),
                        document_id=document_id,
                        document_name=page.document_name,
                        page_number=page.page_number,
                        content=body,
                        content_type=ContentType.TEXT,
                        char_count=len(body),
                        chunk_index=index,
                    )
                )
                index += 1

        # 2. tables - never split, they are only useful whole
        for table in page.tables:
            body = table.markdown or ""
            if len(body) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=table.table_id,
                    document_id=document_id,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    content="Table on page " + str(page.page_number) + ":\n" + body[: size * 3],
                    content_type=ContentType.TABLE,
                    char_count=len(body),
                    chunk_index=index,
                )
            )
            index += 1

        # 3. figure captions - what makes "find Figure 3" work
        for image in page.images:
            if len(image.caption or "") < 8:
                continue
            body = image.caption + " (figure on page " + str(page.page_number) + ")"
            chunks.append(
                Chunk(
                    chunk_id=image.image_id + "_cap",
                    document_id=document_id,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    content=body,
                    # the caption chunk carries the path to the actual picture,
                    # which is what lets a visual question surface the image
                    image_path=image.path,
                    content_type=ContentType.FIGURE,
                    char_count=len(body),
                    chunk_index=index,
                )
            )
            index += 1

    log.info("%d chunks created (size=%d overlap=%d)", len(chunks), size, overlap)
    return chunks
