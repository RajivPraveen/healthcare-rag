"""Shared data contracts that flow through the pipeline.

Document -> Chunk -> ScoredChunk -> Answer(+Citation)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["research_paper", "drug_label", "guideline", "standard", "other"]


class Block(BaseModel):
    """A contiguous run of text that carries its own page + section provenance.

    Every loader normalises to blocks so that a PDF page, a JATS <sec>, and an
    openFDA label field all reach the chunker in the same shape.
    """

    page: int
    section: str
    text: str


class Document(BaseModel):
    """A single source document after text extraction."""

    document_id: str
    title: str
    source: str
    source_type: SourceType = "other"
    blocks: list[Block] = Field(default_factory=list)
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    published: str | None = None
    publisher: str | None = None
    license: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def n_pages(self) -> int:
        return max((b.page for b in self.blocks), default=0)

    @property
    def sections(self) -> list[str]:
        seen: list[str] = []
        for b in self.blocks:
            if b.section not in seen:
                seen.append(b.section)
        return seen

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    def page_text(self, page: int) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.page == page)


class Chunk(BaseModel):
    """A retrievable unit of text plus the metadata needed to cite it."""

    chunk_id: str
    document_id: str
    title: str
    source: str
    source_type: SourceType = "other"
    page: int
    section: str = "Body"
    text: str
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def citation_label(self) -> str:
        return f"{self.title} — p.{self.page} — {self.section}"


class ScoredChunk(BaseModel):
    """A chunk with the scores attached by each retrieval stage."""

    chunk: Chunk
    score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    rank: int | None = None
    retriever: str = "unknown"


class Citation(BaseModel):
    """A source the model actually referenced, as shown in the UI."""

    marker: int
    document_id: str
    document: str
    title: str
    page: int
    section: str
    snippet: str
    url: str | None = None
    score: float | None = None


class Timings(BaseModel):
    retrieval_latency: float = 0.0
    rerank_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class Answer(BaseModel):
    """The full, citation-grounded response returned by the RAG pipeline."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[ScoredChunk] = Field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = True
    strategy: str = "hybrid+rerank"
    model: str = ""
    timings: Timings = Field(default_factory=Timings)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class EvalQuestion(BaseModel):
    """One labelled item in the evaluation set."""

    question_id: str
    question: str
    expected_information: str
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    must_include_terms: list[str] = Field(default_factory=list)
    category: str = "general"
    source_title: str | None = None
    # Fraction of question terms also present in the gold chunk. High values
    # mean the question leaks its answer's vocabulary and flatters lexical search.
    lexical_overlap: float = 0.0
    unanswerable: bool = False
