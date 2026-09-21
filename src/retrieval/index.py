"""The on-disk artefact that retrieval runs against.

A ``RagIndex`` bundles everything a query needs: the chunks themselves, the
dense vector store, and the BM25 inverted index. Keeping them in one directory
means an experiment ("500-token chunks, bge-small") is a single reproducible
artefact you can rebuild, diff, or ship into a container.

    data/indexes/<name>/
        chunks.jsonl     every chunk with full citation metadata
        faiss.index      dense vectors
        bm25.pkl         inverted index
        manifest.json    model + chunk config + corpus stats
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import Chunk
from src.embeddings.embedder import Embedder, get_embedder
from src.ingestion.chunker import ChunkConfig, contextual_text
from src.retrieval.bm25 import BM25Index
from src.retrieval.vector_search import VectorStore, make_vector_store

log = get_logger(__name__)

DEFAULT_INDEX_NAME = "default"


class RagIndex:
    def __init__(self, name: str = DEFAULT_INDEX_NAME) -> None:
        self.name = name
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self.vector_store: VectorStore | None = None
        self.bm25: BM25Index | None = None
        self.manifest: dict[str, Any] = {}

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        name: str = DEFAULT_INDEX_NAME,
        embedder: Embedder | None = None,
        backend: str | None = None,
        chunk_config: ChunkConfig | None = None,
        with_vectors: bool = True,
        with_bm25: bool = True,
    ) -> RagIndex:
        index = cls(name=name)
        index.chunks = chunks
        index.by_id = {c.chunk_id: c for c in chunks}

        embedder = embedder or get_embedder()

        if with_vectors:
            texts = [contextual_text(c) for c in chunks]
            vectors = embedder.encode_passages(texts)
            embedder.save_cache()
            index.vector_store = make_vector_store(backend, dimension=vectors.shape[1])
            index.vector_store.build(chunks, vectors)

        if with_bm25:
            index.bm25 = BM25Index()
            index.bm25.build(chunks)

        n_docs = len({c.document_id for c in chunks})
        index.manifest = {
            "name": name,
            "built_at": datetime.now(UTC).isoformat(),
            "embedding_model": embedder.model_name,
            "vector_backend": backend or get_settings().vector_backend,
            "chunk_config": asdict(chunk_config) if chunk_config else None,
            "n_chunks": len(chunks),
            "n_documents": n_docs,
            "mean_tokens": (
                float(np.mean([c.token_count for c in chunks])) if chunks else 0.0
            ),
        }
        return index

    # -- persistence ---------------------------------------------------------

    def path(self) -> Path:
        return get_settings().index_dir / self.name

    def save(self, path: Path | None = None) -> Path:
        path = path or self.path()
        path.mkdir(parents=True, exist_ok=True)

        with (path / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")

        if self.vector_store is not None:
            self.vector_store.save(path)
        if self.bm25 is not None:
            self.bm25.save(path)

        (path / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        log.info("index '%s' saved to %s", self.name, path)
        return path

    @classmethod
    def load(cls, name: str = DEFAULT_INDEX_NAME, path: Path | None = None) -> RagIndex:
        path = path or (get_settings().index_dir / name)
        if not path.exists():
            raise FileNotFoundError(
                f"No index at {path}. Build one first: `hcrag index`"
            )

        index = cls(name=name)
        index.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

        with (path / "chunks.jsonl").open(encoding="utf-8") as fh:
            index.chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        index.by_id = {c.chunk_id: c for c in index.chunks}

        backend = index.manifest.get("vector_backend", "faiss")
        if (path / "faiss.index").exists() or (path / "qdrant.json").exists():
            store = make_vector_store(backend)
            store.load(path)
            index.vector_store = store

        if (path / "bm25.pkl").exists():
            bm25 = BM25Index()
            bm25.load(path)
            index.bm25 = bm25

        log.info(
            "index '%s' loaded: %d chunks from %d documents",
            name,
            len(index.chunks),
            index.manifest.get("n_documents", 0),
        )
        return index

    # -- access --------------------------------------------------------------

    def get(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)

    def documents(self) -> list[dict[str, Any]]:
        """One row per source document, for the /documents endpoint and the UI."""
        seen: dict[str, dict[str, Any]] = {}
        for chunk in self.chunks:
            entry = seen.setdefault(
                chunk.document_id,
                {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "source_type": chunk.source_type,
                    "url": chunk.url,
                    "n_chunks": 0,
                    "pages": 0,
                    "sections": set(),
                },
            )
            entry["n_chunks"] += 1
            entry["pages"] = max(entry["pages"], chunk.page)
            entry["sections"].add(chunk.section)
        out = []
        for entry in seen.values():
            entry["sections"] = sorted(entry["sections"])
            entry["n_sections"] = len(entry["sections"])
            out.append(entry)
        return sorted(out, key=lambda e: e["title"])

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)
