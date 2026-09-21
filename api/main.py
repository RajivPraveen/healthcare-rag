"""Module 11 - FastAPI backend.

    POST /query          ask a question, get a grounded answer + citations
    POST /retrieve       retrieval only (no LLM) - useful for debugging relevance
    POST /documents      upload a PDF into the corpus
    GET  /documents      list indexed documents
    GET  /documents/{id} document detail with its chunks
    GET  /health         liveness + component status
    GET  /metrics        rolling service metrics
    GET  /evaluation     the most recent evaluation results

The index and both local models are loaded once during lifespan startup and
warmed, so request latency reflects actual work rather than lazy model loading.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.common.config import get_settings
from src.common.logging import get_logger, setup_logging
from src.common.schemas import Answer
from src.generation.pipeline import RAGConfig, RAGPipeline
from src.retrieval.index import RagIndex

setup_logging()
log = get_logger(__name__)

# Mutable module state, populated at startup.
STATE: dict[str, Any] = {
    "index": None,
    "pipelines": {},
    "started_at": None,
    "latencies": deque(maxlen=500),
    "queries": 0,
    "errors": 0,
    "tokens": 0,
    "cost": 0.0,
}


def _pipeline(strategy: str, rerank: bool, top_k: int, top_n: int) -> RAGPipeline:
    """Cache one pipeline per configuration so models are shared, not reloaded."""
    key = (strategy, rerank, top_k, top_n)
    if key not in STATE["pipelines"]:
        config = RAGConfig(strategy=strategy, rerank=rerank, top_k=top_k, top_n=top_n)
        STATE["pipelines"][key] = RAGPipeline(STATE["index"], config=config)
    return STATE["pipelines"][key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("loading index…")
    try:
        STATE["index"] = RagIndex.load("default")
    except FileNotFoundError:
        log.error("No index found. Run `hcrag ingest && hcrag index` before serving.")
        STATE["index"] = None

    if STATE["index"] is not None:
        default = _pipeline("hybrid", True, settings.retrieval_top_k, settings.rerank_top_n)
        default.warmup()

    STATE["started_at"] = time.time()
    yield
    log.info("shutting down")


app = FastAPI(
    title="Healthcare RAG Intelligence Platform",
    description="Grounded clinical question answering with citations over medical literature.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What are the first-line treatments?"])
    strategy: Literal["vector", "bm25", "hybrid", "hybrid_weighted"] = "hybrid"
    rerank: bool = True
    top_k: int = Field(20, ge=1, le=100)
    top_n: int = Field(5, ge=1, le=20)


class SourceOut(BaseModel):
    marker: int
    document: str
    document_id: str
    title: str
    page: int
    section: str
    snippet: str
    url: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    confidence: float
    grounded: bool
    strategy: str
    model: str
    retrieval_latency: float
    rerank_latency: float
    generation_latency: float
    total_latency: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


def _to_response(result: Answer) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceOut(
                marker=c.marker,
                document=c.document,
                document_id=c.document_id,
                title=c.title,
                page=c.page,
                section=c.section,
                snippet=c.snippet,
                url=c.url,
                score=c.score,
            )
            for c in result.citations
        ],
        confidence=result.confidence,
        grounded=result.grounded,
        strategy=result.strategy,
        model=result.model,
        retrieval_latency=result.timings.retrieval_latency,
        rerank_latency=result.timings.rerank_latency,
        generation_latency=result.timings.generation_latency,
        total_latency=result.timings.total_latency,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        estimated_cost_usd=result.usage.estimated_cost_usd,
    )


def _require_index() -> RagIndex:
    if STATE["index"] is None:
        raise HTTPException(
            status_code=503,
            detail="Index not built. Run `hcrag ingest && hcrag index`, then restart.",
        )
    return STATE["index"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(request: QueryRequest) -> QueryResponse:
    """Ask a clinical question and receive an answer grounded in the corpus."""
    _require_index()
    try:
        pipeline = _pipeline(request.strategy, request.rerank, request.top_k, request.top_n)
        result = pipeline.answer(request.question)
    except Exception as exc:
        STATE["errors"] += 1
        log.exception("query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    STATE["queries"] += 1
    STATE["latencies"].append(result.timings.total_latency)
    STATE["tokens"] += result.usage.total_tokens
    STATE["cost"] += result.usage.estimated_cost_usd
    return _to_response(result)


class RetrieveResponse(BaseModel):
    query: str
    strategy: str
    latency: float
    results: list[dict[str, Any]]


@app.post("/retrieve", response_model=RetrieveResponse, tags=["rag"])
def retrieve(request: QueryRequest) -> RetrieveResponse:
    """Retrieval only, no generation. Lets you inspect ranking without LLM cost."""
    _require_index()
    pipeline = _pipeline(request.strategy, request.rerank, request.top_k, request.top_n)
    started = time.perf_counter()
    contexts, retrieval_latency, rerank_latency = pipeline.retrieve(request.question)
    return RetrieveResponse(
        query=request.question,
        strategy=pipeline.config.label,
        latency=round(time.perf_counter() - started, 4),
        results=[
            {
                "rank": c.rank,
                "chunk_id": c.chunk.chunk_id,
                "document_id": c.chunk.document_id,
                "title": c.chunk.title,
                "page": c.chunk.page,
                "section": c.chunk.section,
                "score": round(c.score, 4),
                "vector_score": c.vector_score,
                "bm25_score": c.bm25_score,
                "rerank_score": c.rerank_score,
                "text": c.chunk.text[:600],
            }
            for c in contexts
        ],
    )


@app.get("/documents", tags=["corpus"])
def list_documents(
    source_type: str | None = None, limit: int = 500, offset: int = 0
) -> dict[str, Any]:
    index = _require_index()
    docs = index.documents()
    if source_type:
        docs = [d for d in docs if d["source_type"] == source_type]
    return {"total": len(docs), "documents": docs[offset : offset + limit]}


@app.get("/documents/{document_id}", tags=["corpus"])
def get_document(document_id: str, include_chunks: bool = False) -> dict[str, Any]:
    index = _require_index()
    chunks = [c for c in index.chunks if c.document_id == document_id]
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No document '{document_id}'")

    first = chunks[0]
    payload: dict[str, Any] = {
        "document_id": document_id,
        "title": first.title,
        "source": first.source,
        "source_type": first.source_type,
        "url": first.url,
        "n_chunks": len(chunks),
        "n_pages": max(c.page for c in chunks),
        "sections": sorted({c.section for c in chunks}),
    }
    if include_chunks:
        payload["chunks"] = [
            {
                "chunk_id": c.chunk_id,
                "page": c.page,
                "section": c.section,
                "tokens": c.token_count,
                "text": c.text,
            }
            for c in chunks
        ]
    return payload


def _reindex() -> None:
    """Rebuild the index in the background after an upload."""
    from src.ingestion.chunker import ChunkConfig, chunk_documents
    from src.ingestion.loader import iter_documents

    settings = get_settings()
    try:
        docs = list(iter_documents(settings.raw_dir))
        chunks = chunk_documents(
            docs,
            ChunkConfig(
                target_tokens=settings.chunk_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            ),
        )
        index = RagIndex.build(chunks, name="default")
        index.save()
        STATE["index"] = index
        STATE["pipelines"].clear()
        log.info("reindex complete: %d chunks", len(chunks))
    except Exception:
        log.exception("reindex failed")


@app.post("/documents", tags=["corpus"], status_code=202)
async def upload_document(file: UploadFile, background: BackgroundTasks) -> dict[str, str]:
    """Add a PDF to the corpus. Reindexing happens in the background."""
    if not file.filename or not file.filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .pdf, .txt and .md are supported")

    settings = get_settings()
    target_dir = settings.raw_dir / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(file.filename).name

    with target.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    background.add_task(_reindex)
    return {
        "status": "accepted",
        "filename": target.name,
        "detail": "File stored; the index is rebuilding in the background.",
    }


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    settings = get_settings()
    index = STATE["index"]
    from src.generation.llm import get_llm

    llm = get_llm()
    return {
        "status": "ok" if index is not None else "degraded",
        "index_loaded": index is not None,
        "n_chunks": index.n_chunks if index else 0,
        "n_documents": index.manifest.get("n_documents", 0) if index else 0,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "device": settings.resolve_device(),
        "uptime_seconds": round(time.time() - (STATE["started_at"] or time.time()), 1),
    }


@app.get("/metrics", tags=["ops"])
def metrics() -> dict[str, Any]:
    latencies = sorted(STATE["latencies"])
    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        return round(latencies[min(int(p * (len(latencies) - 1)), len(latencies) - 1)], 4)

    return {
        "queries_total": STATE["queries"],
        "errors_total": STATE["errors"],
        "tokens_total": STATE["tokens"],
        "estimated_cost_usd": round(STATE["cost"], 6),
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
        },
    }


@app.get("/evaluation", tags=["ops"])
def evaluation() -> dict[str, Any]:
    """Serve the most recent evaluation artefacts to the dashboard."""
    results_dir = get_settings().eval_dir / "results"
    payload: dict[str, Any] = {"strategy_comparison": None, "chunk_sweep": None, "latest_run": None}

    comparison = results_dir / "strategy_comparison.json"
    if comparison.exists():
        payload["strategy_comparison"] = json.loads(comparison.read_text())

    sweep = results_dir / "chunk_sweep.json"
    if sweep.exists():
        payload["chunk_sweep"] = json.loads(sweep.read_text())

    runs = sorted(results_dir.glob("*_2*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if runs:
        payload["latest_run"] = json.loads(runs[0].read_text())

    return payload


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "Healthcare RAG Intelligence Platform",
        "docs": "/docs",
        "health": "/health",
    }
