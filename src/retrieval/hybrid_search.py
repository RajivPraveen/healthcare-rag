"""Module 5 - the three retrieval strategies behind one interface.

Version A  ``VectorRetriever``  dense semantic search
Version B  ``BM25Retriever``    lexical search
Version C  ``HybridRetriever``  fusion of both

Two fusion methods are provided because they fail differently:

* **Reciprocal Rank Fusion** combines *ranks*, not scores. Cosine similarity
  (bounded, ~0.6-0.9) and BM25 (unbounded, corpus-dependent) live on
  incomparable scales, and RRF sidesteps that entirely. It is the robust default.
* **Weighted score fusion** min-max normalises each arm and blends with
  ``alpha``. More tunable, but normalisation is computed per-query over the
  candidate pool, so it is sensitive to pool composition.

``alpha`` and the fusion method are constructor arguments precisely so
Module 9 can measure them rather than assert them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import ScoredChunk
from src.embeddings.embedder import Embedder, get_embedder
from src.retrieval.index import RagIndex
from src.retrieval.query_processing import expand_query

log = get_logger(__name__)

RRF_K = 60  # standard damping constant from Cormack et al.


class Retriever(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]: ...

    def _materialise(
        self, index: RagIndex, hits: list[tuple[str, float]], retriever: str
    ) -> list[ScoredChunk]:
        out: list[ScoredChunk] = []
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            chunk = index.get(chunk_id)
            if chunk is None:
                continue
            out.append(ScoredChunk(chunk=chunk, score=score, rank=rank, retriever=retriever))
        return out


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(
        self,
        index: RagIndex,
        embedder: Embedder | None = None,
        top_k: int | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder or get_embedder(index.manifest.get("embedding_model"))
        self.top_k = top_k or get_settings().retrieval_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        if self.index.vector_store is None:
            raise RuntimeError("index has no vector store")
        k = top_k or self.top_k
        qvec = self.embedder.encode_query(query)
        hits = self.index.vector_store.search(qvec, top_k=k)
        scored = self._materialise(self.index, hits, "vector")
        for s in scored:
            s.vector_score = s.score
        return scored


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index: RagIndex, top_k: int | None = None, expand: bool = True) -> None:
        self.index = index
        self.top_k = top_k or get_settings().retrieval_top_k
        self.expand = expand

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        if self.index.bm25 is None:
            raise RuntimeError("index has no BM25 index")
        k = top_k or self.top_k
        # Abbreviation expansion matters far more for lexical than dense search:
        # "MI" and "myocardial infarction" share no tokens at all.
        text = expand_query(query) if self.expand else query
        hits = self.index.bm25.search(text, top_k=k)
        scored = self._materialise(self.index, hits, "bm25")
        for s in scored:
            s.bm25_score = s.score
        return scored


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(
        self,
        index: RagIndex,
        embedder: Embedder | None = None,
        top_k: int | None = None,
        fusion: str = "rrf",
        alpha: float | None = None,
        candidate_multiplier: int = 2,
    ) -> None:
        settings = get_settings()
        self.index = index
        self.top_k = top_k or settings.retrieval_top_k
        self.fusion = fusion
        self.alpha = settings.hybrid_alpha if alpha is None else alpha
        # Each arm retrieves deeper than the final k so fusion has room to work.
        self.candidate_multiplier = candidate_multiplier
        self.vector = VectorRetriever(index, embedder=embedder, top_k=self.top_k)
        self.bm25 = BM25Retriever(index, top_k=self.top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or self.top_k
        pool = k * self.candidate_multiplier

        vector_hits = self.vector.retrieve(query, top_k=pool)
        bm25_hits = self.bm25.retrieve(query, top_k=pool)

        merged: dict[str, ScoredChunk] = {}

        if self.fusion == "rrf":
            fused: dict[str, float] = {}
            for hits in (vector_hits, bm25_hits):
                for rank, hit in enumerate(hits, start=1):
                    cid = hit.chunk.chunk_id
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                    entry = merged.setdefault(
                        cid, ScoredChunk(chunk=hit.chunk, score=0.0, retriever="hybrid")
                    )
                    if hit.vector_score is not None:
                        entry.vector_score = hit.vector_score
                    if hit.bm25_score is not None:
                        entry.bm25_score = hit.bm25_score
            for cid, score in fused.items():
                merged[cid].score = score

        elif self.fusion == "weighted":
            vec_norm = dict(
                zip(
                    [h.chunk.chunk_id for h in vector_hits],
                    _minmax([h.score for h in vector_hits]),
                )
            )
            bm_norm = dict(
                zip(
                    [h.chunk.chunk_id for h in bm25_hits],
                    _minmax([h.score for h in bm25_hits]),
                )
            )
            for hits in (vector_hits, bm25_hits):
                for hit in hits:
                    cid = hit.chunk.chunk_id
                    entry = merged.setdefault(
                        cid, ScoredChunk(chunk=hit.chunk, score=0.0, retriever="hybrid")
                    )
                    if hit.vector_score is not None:
                        entry.vector_score = hit.vector_score
                    if hit.bm25_score is not None:
                        entry.bm25_score = hit.bm25_score
            for cid, entry in merged.items():
                entry.score = self.alpha * vec_norm.get(cid, 0.0) + (1 - self.alpha) * bm_norm.get(
                    cid, 0.0
                )
        else:
            raise ValueError(f"unknown fusion method: {self.fusion!r}")

        ranked = sorted(merged.values(), key=lambda s: s.score, reverse=True)[:k]
        for rank, item in enumerate(ranked, start=1):
            item.rank = rank
        return ranked


def make_retriever(
    strategy: str,
    index: RagIndex,
    embedder: Embedder | None = None,
    top_k: int | None = None,
    **kwargs,
) -> Retriever:
    strategy = strategy.lower()
    if strategy == "vector":
        return VectorRetriever(index, embedder=embedder, top_k=top_k)
    if strategy == "bm25":
        return BM25Retriever(index, top_k=top_k)
    if strategy in {"hybrid", "hybrid_rrf"}:
        return HybridRetriever(index, embedder=embedder, top_k=top_k, fusion="rrf", **kwargs)
    if strategy == "hybrid_weighted":
        return HybridRetriever(index, embedder=embedder, top_k=top_k, fusion="weighted", **kwargs)
    raise ValueError(f"unknown retrieval strategy: {strategy!r}")


def score_matrix(hits: list[ScoredChunk]) -> np.ndarray:
    """Helper for notebooks: (n, 3) array of vector / bm25 / fused scores."""
    return np.array(
        [[h.vector_score or 0.0, h.bm25_score or 0.0, h.score] for h in hits], dtype=np.float32
    )
