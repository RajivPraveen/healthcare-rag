"""Module 2 (part 3) + Module 3 - chunking.

Chunking is a tunable knob, not a constant, so the whole strategy is driven by
``ChunkConfig``. That is what lets ``notebooks/02_chunking_experiment.ipynb``
sweep 300/500/800/1200 tokens and measure the retrieval impact.

Design decisions worth knowing:

* Chunks respect **sentence** boundaries, so a dosage or contraindication is
  never cut in half.
* Chunks never cross a **section** boundary. "Contraindications" text bleeding
  into "Dosage" would produce a citation that points at the wrong clinical claim.
* Chunks carry an **overlap** tail so a fact spanning a boundary stays retrievable.
* ``contextual_text`` prepends title/section when embedding. A chunk reading
  "Do not exceed 40 mg daily" is meaningless in isolation; the document and
  section give the embedding the subject it otherwise lacks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.common.schemas import Block, Chunk, Document
from src.common.text import count_tokens, split_sentences, truncate_tokens


@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int = 500
    overlap_tokens: int = 75
    min_tokens: int = 50
    # Hard ceiling so a single monster sentence can't blow the LLM context.
    max_tokens: int = 2000
    respect_sections: bool = True

    @property
    def name(self) -> str:
        return f"t{self.target_tokens}_o{self.overlap_tokens}"


def _chunk_id(document_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{document_id}::{index:04d}::{digest}"


def _group_blocks(doc: Document, respect_sections: bool) -> list[list[Block]]:
    """Merge consecutive blocks so a chunk can span pages inside one section."""
    if not doc.blocks:
        return []
    if not respect_sections:
        return [list(doc.blocks)]

    groups: list[list[Block]] = []
    current: list[Block] = [doc.blocks[0]]
    for block in doc.blocks[1:]:
        if block.section == current[-1].section:
            current.append(block)
        else:
            groups.append(current)
            current = [block]
    groups.append(current)
    return groups


def _sentences_with_pages(blocks: list[Block]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for block in blocks:
        for sentence in split_sentences(block.text):
            out.append((sentence, block.page))
    return out


def _overlap_tail(
    sentences: list[tuple[str, int]], token_counts: list[int], overlap_tokens: int
) -> tuple[list[tuple[str, int]], list[int]]:
    """Take sentences from the end of a chunk until we've covered the overlap budget."""
    if overlap_tokens <= 0:
        return [], []
    tail: list[tuple[str, int]] = []
    tail_counts: list[int] = []
    total = 0
    for sent, count in zip(reversed(sentences), reversed(token_counts)):
        if total >= overlap_tokens:
            break
        tail.insert(0, sent)
        tail_counts.insert(0, count)
        total += count
    # Never let the overlap become the entire chunk.
    if len(tail) >= len(sentences):
        tail, tail_counts = tail[1:], tail_counts[1:]
    return tail, tail_counts


def chunk_document(doc: Document, config: ChunkConfig | None = None) -> list[Chunk]:
    config = config or ChunkConfig()
    chunks: list[Chunk] = []
    index = 0
    char_cursor = 0

    for group in _group_blocks(doc, config.respect_sections):
        section = group[0].section
        sentences = _sentences_with_pages(group)
        if not sentences:
            continue

        token_counts = [count_tokens(s) for s, _ in sentences]

        buffer: list[tuple[str, int]] = []
        buffer_counts: list[int] = []

        # `section` is passed in rather than captured: the closure is
        # redefined each iteration, so capturing it would silently attach the
        # wrong section name to every chunk if flush were ever deferred.
        def flush(section: str) -> None:
            nonlocal buffer, buffer_counts, index, char_cursor
            if not buffer:
                return
            text = " ".join(s for s, _ in buffer).strip()
            n_tokens = sum(buffer_counts)
            if n_tokens > config.max_tokens:
                text = truncate_tokens(text, config.max_tokens)
                n_tokens = config.max_tokens
            # Drop slivers, unless the section produced nothing else at all.
            if n_tokens < config.min_tokens:
                buffer, buffer_counts = [], []
                return

            start_page = buffer[0][1]
            end_page = buffer[-1][1]
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.document_id, index, text),
                    document_id=doc.document_id,
                    title=doc.title,
                    source=doc.source,
                    source_type=doc.source_type,
                    page=start_page,
                    section=section,
                    text=text,
                    token_count=n_tokens,
                    char_start=char_cursor,
                    char_end=char_cursor + len(text),
                    url=doc.url,
                    extra={
                        "page_end": end_page,
                        "published": doc.published,
                        "publisher": doc.publisher,
                        "chunk_config": config.name,
                    },
                )
            )
            index += 1
            char_cursor += len(text)
            buffer, buffer_counts = [], []

        for (sentence, page), n_tokens in zip(sentences, token_counts):
            if buffer and sum(buffer_counts) + n_tokens > config.target_tokens:
                prev, prev_counts = list(buffer), list(buffer_counts)
                flush(section)
                tail, tail_counts = _overlap_tail(prev, prev_counts, config.overlap_tokens)
                buffer, buffer_counts = tail, tail_counts
            buffer.append((sentence, page))
            buffer_counts.append(n_tokens)

        flush(section)

    return chunks


def chunk_documents(docs: list[Document], config: ChunkConfig | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, config))
    return out


def contextual_text(chunk: Chunk) -> str:
    """The string actually fed to the embedding model.

    Prefixing document title and section turns an anaphoric fragment into a
    self-contained passage, which measurably lifts recall on short chunks.
    """
    return f"{chunk.title}\n{chunk.section}\n\n{chunk.text}"
