"""Module 5, Version A - dense vector retrieval.

Two interchangeable backends behind one interface:

* ``FaissStore``  - a local flat inner-product index. Because embeddings are
  L2-normalised, inner product *is* cosine similarity, and a flat index gives
  exact search. At corpus scale (thousands of chunks) exact search costs
  ~1 ms, so approximate indexes would trade recall for nothing.
* ``QdrantStore`` - the same API against a Qdrant server, for the Docker
  deployment and for payload-filtered search.
"""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import Chunk

log = get_logger(__name__)


class VectorStore(ABC):
    """Minimal contract every vector backend implements."""

    @abstractmethod
    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, top_k: int = 20, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, similarity) pairs, best first."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...

    @property
    @abstractmethod
    def size(self) -> int: ...


_FAISS = None


def _import_faiss():
    """Import faiss safely alongside torch.

    faiss-cpu and torch each bundle their own OpenMP runtime. Two things go
    wrong when both are live in one process on macOS:

    1. Whichever loads second aborts at import ("libomp already initialised").
       Importing torch first makes the order deterministic, and
       ``KMP_DUPLICATE_LIB_OK`` in ``src/__init__`` lets the duplicate load.
    2. More subtly, a *parallel* faiss search after torch has run segfaults —
       and only silence comes back, no traceback. That one cost real debugging
       time: the chunk sweep died inside ``index.search()`` on a freshly built
       index while the identical call on a disk-loaded index was fine.

    Pinning faiss to one thread sidesteps the shared-runtime conflict
    entirely. For the flat indexes used here (tens of thousands of vectors,
    sub-millisecond searches) the lost parallelism is not measurable.
    """
    global _FAISS
    if _FAISS is not None:
        return _FAISS

    # torch must initialise first. Loaded via importlib so that neither the
    # import sorter nor a well-meaning cleanup can reorder these two lines,
    # which would reintroduce the abort described above.
    importlib.import_module("torch")
    faiss = importlib.import_module("faiss")

    faiss.omp_set_num_threads(1)
    _FAISS = faiss
    return faiss


class FaissStore(VectorStore):
    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension
        self.index = None
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        faiss = _import_faiss()

        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError(
                f"vectors {vectors.shape} do not line up with {len(chunks)} chunks"
            )
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.dimension = vectors.shape[1]
        # Vectors are already normalised, so IP == cosine.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.chunk_ids = [c.chunk_id for c in chunks]
        log.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, self.dimension)

    def search(
        self, query_vector: np.ndarray, top_k: int = 20, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        _import_faiss()  # ensure the thread pinning above is applied
        q = np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32))
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        return [
            (self.chunk_ids[i], float(s))
            for i, s in zip(indices[0], scores[0])
            if i != -1
        ]

    def save(self, path: Path) -> None:
        faiss = _import_faiss()

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        (path / "faiss_ids.json").write_text(
            json.dumps({"dimension": self.dimension, "chunk_ids": self.chunk_ids}),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        faiss = _import_faiss()

        self.index = faiss.read_index(str(path / "faiss.index"))
        payload = json.loads((path / "faiss_ids.json").read_text(encoding="utf-8"))
        self.dimension = payload["dimension"]
        self.chunk_ids = payload["chunk_ids"]

    @property
    def size(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)


class QdrantStore(VectorStore):
    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        dimension: int | None = None,
    ) -> None:
        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.dimension = dimension
        self._client = None
        self._id_map: dict[int, str] = {}

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, timeout=60)
        return self._client

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.dimension = vectors.shape[1]
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )

        points: list[PointStruct] = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            self._id_map[i] = chunk.chunk_id
            points.append(
                PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "source": chunk.source,
                        "source_type": chunk.source_type,
                        "page": chunk.page,
                        "section": chunk.section,
                    },
                )
            )

        for start in range(0, len(points), 256):
            self.client.upsert(collection_name=self.collection, points=points[start : start + 256])
        log.info("Qdrant collection %s built: %d points", self.collection, len(points))

    def search(
        self, query_vector: np.ndarray, top_k: int = 20, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qfilter = None
        if filters:
            qfilter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()
                ]
            )

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [(h.payload["chunk_id"], float(h.score)) for h in hits]

    def save(self, path: Path) -> None:
        # Qdrant persists server-side; record enough to reconnect.
        path.mkdir(parents=True, exist_ok=True)
        (path / "qdrant.json").write_text(
            json.dumps(
                {"url": self.url, "collection": self.collection, "dimension": self.dimension}
            ),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        cfg = json.loads((path / "qdrant.json").read_text(encoding="utf-8"))
        self.url = cfg["url"]
        self.collection = cfg["collection"]
        self.dimension = cfg["dimension"]

    @property
    def size(self) -> int:
        try:
            return int(self.client.count(self.collection).count)
        except Exception:
            return 0


def make_vector_store(backend: str | None = None, dimension: int | None = None) -> VectorStore:
    backend = (backend or get_settings().vector_backend).lower()
    if backend == "faiss":
        return FaissStore(dimension=dimension)
    if backend == "qdrant":
        return QdrantStore(dimension=dimension)
    raise ValueError(f"unknown vector backend: {backend!r}")
