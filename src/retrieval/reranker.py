"""Module 6 - cross-encoder reranking.

The retrieval arms are *bi-encoders*: query and passage are embedded
independently, so the model never sees them together and similarity is a
single dot product. That is what makes retrieval over thousands of chunks
fast, and also what caps its precision.

A *cross-encoder* concatenates [query, passage] into one input and runs full
attention across the pair, so it can judge whether the passage actually
answers the question rather than merely sharing a topic. It is far too slow to
run over the corpus (one forward pass per pair), which is exactly why it is a
second stage: retrieve 20 cheaply, rerank to 5 precisely.

This is the single highest-leverage precision win in the pipeline, and because
only the top 5 reach the LLM it also cuts prompt tokens and cost.
"""

from __future__ import annotations

import threading

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import ScoredChunk

log = get_logger(__name__)


class Reranker:
    """No-op baseline. Keeps the ablation ("no reranker") a first-class config."""

    name = "none"

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int | None = None
    ) -> list[ScoredChunk]:
        n = top_n or len(candidates)
        return candidates[:n]


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        max_per_document: int = 2,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.resolve_device()
        self.batch_size = batch_size
        self.max_length = max_length
        # Chunk overlap plus a long relevant section means one document can
        # legitimately occupy every top slot. That wastes context on
        # near-duplicate text and produces an answer citing a single source,
        # so cap each document's share and let the next document through.
        self.max_per_document = max_per_document
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    log.info("loading reranker %s on %s", self.model_name, self.device)
                    self._model = CrossEncoder(
                        self.model_name, device=self.device, max_length=self.max_length
                    )
        return self._model

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int | None = None
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        n = top_n or get_settings().rerank_top_n

        # Give the cross-encoder the same contextual framing the embedder saw,
        # so a passage isn't penalised for lacking its own subject.
        pairs = [
            (query, f"{c.chunk.title} | {c.chunk.section}\n{c.chunk.text}") for c in candidates
        ]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        ranked = sorted(candidates, key=lambda c: c.rerank_score or float("-inf"), reverse=True)
        top = self._diversify(ranked, n)
        for rank, item in enumerate(top, start=1):
            item.rank = rank
            # Downstream consumers read `.score`; make it the authoritative one.
            item.score = item.rerank_score if item.rerank_score is not None else item.score
            item.retriever = f"{item.retriever}+rerank"
        return top

    def _diversify(self, ranked: list[ScoredChunk], n: int) -> list[ScoredChunk]:
        """Greedily take the best passages under a per-document quota.

        If the quota leaves us short (e.g. only one document is relevant at
        all), the remaining slots are backfilled in score order rather than
        returning fewer passages than asked for.
        """
        if self.max_per_document <= 0:
            return ranked[:n]

        selected: list[ScoredChunk] = []
        per_doc: dict[str, int] = {}
        overflow: list[ScoredChunk] = []

        for candidate in ranked:
            doc_id = candidate.chunk.document_id
            if per_doc.get(doc_id, 0) < self.max_per_document:
                selected.append(candidate)
                per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
                if len(selected) == n:
                    return selected
            else:
                overflow.append(candidate)

        for candidate in overflow:
            if len(selected) == n:
                break
            selected.append(candidate)
        return selected


_DEFAULT: CrossEncoderReranker | None = None


def get_reranker(model_name: str | None = None) -> CrossEncoderReranker:
    global _DEFAULT
    if _DEFAULT is None or (model_name and model_name != _DEFAULT.model_name):
        _DEFAULT = CrossEncoderReranker(model_name=model_name)
    return _DEFAULT


def make_reranker(kind: str = "cross-encoder") -> Reranker:
    if kind in {"none", "noop", ""}:
        return Reranker()
    return get_reranker()
