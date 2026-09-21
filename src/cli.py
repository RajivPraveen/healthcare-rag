"""Command-line entry point: `hcrag <command>`.

    hcrag ingest      download the healthcare corpus
    hcrag process     extract, clean and chunk into data/processed
    hcrag index       build the FAISS + BM25 indexes
    hcrag ask         ask a question from the terminal
    hcrag evaluate    run the RAG evaluation harness
    hcrag compare     compare retrieval strategies head to head
    hcrag sweep       run the chunk-size experiment
    hcrag info        show corpus / index / provider status
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.common.config import get_settings
from src.common.logging import setup_logging

app = typer.Typer(
    add_completion=False,
    help="Healthcare RAG Intelligence Platform",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    setup_logging()


# ---------------------------------------------------------------------------


@app.command()
def ingest(
    per_topic: int = typer.Option(6, help="PMC articles to fetch per clinical topic"),
    pmc_limit: int = typer.Option(None, help="Hard cap on PMC articles"),
    fda_limit: int = typer.Option(None, help="Hard cap on FDA drug labels"),
) -> None:
    """Module 1 - download the corpus into data/raw."""
    from src.ingestion.downloader import build_corpus

    report = build_corpus(pmc_per_topic=per_topic, pmc_limit=pmc_limit, fda_limit=fda_limit)
    console.print(
        Panel(
            f"PubMed Central papers: [bold]{report.pmc}[/]\n"
            f"openFDA drug labels:   [bold]{report.openfda}[/]\n"
            f"Total documents:       [bold green]{report.total}[/]",
            title="Corpus downloaded",
        )
    )


@app.command()
def process(
    chunk_tokens: int = typer.Option(None, help="Target chunk size in tokens"),
    overlap: int = typer.Option(None, help="Overlap in tokens"),
) -> None:
    """Module 2 - extract, clean and chunk. Writes data/processed/chunks.jsonl."""
    from src.ingestion.chunker import ChunkConfig, chunk_documents
    from src.ingestion.loader import iter_documents

    settings = get_settings()
    cfg = ChunkConfig(
        target_tokens=chunk_tokens or settings.chunk_tokens,
        overlap_tokens=overlap if overlap is not None else settings.chunk_overlap_tokens,
    )

    with console.status("Extracting and cleaning documents…"):
        docs = list(iter_documents(settings.raw_dir))
    with console.status(f"Chunking at {cfg.target_tokens} tokens…"):
        chunks = chunk_documents(docs, cfg)

    out = settings.processed_dir / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")

    docs_out = settings.processed_dir / "documents.jsonl"
    with docs_out.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json() + "\n")

    mean_tokens = sum(c.token_count for c in chunks) / max(len(chunks), 1)
    console.print(
        Panel(
            f"Documents: [bold]{len(docs)}[/]\n"
            f"Chunks:    [bold]{len(chunks)}[/]  (mean {mean_tokens:.0f} tokens)\n"
            f"Written:   {out}",
            title=f"Processed @ {cfg.name}",
        )
    )


@app.command()
def index(
    name: str = typer.Option("default", help="Index name under data/indexes"),
    backend: str = typer.Option(None, help="faiss | qdrant"),
    chunk_tokens: int = typer.Option(None),
    overlap: int = typer.Option(None),
    reprocess: bool = typer.Option(
        False, "--reprocess", help="Re-extract from data/raw instead of reusing chunks.jsonl"
    ),
) -> None:
    """Module 4/5 - build the vector + BM25 indexes."""
    from src.common.schemas import Chunk
    from src.ingestion.chunker import ChunkConfig, chunk_documents
    from src.ingestion.loader import iter_documents
    from src.retrieval.index import RagIndex

    settings = get_settings()
    cfg = ChunkConfig(
        target_tokens=chunk_tokens or settings.chunk_tokens,
        overlap_tokens=overlap if overlap is not None else settings.chunk_overlap_tokens,
    )

    cached = settings.processed_dir / "chunks.jsonl"
    if cached.exists() and not reprocess and chunk_tokens is None:
        with cached.open(encoding="utf-8") as fh:
            chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        console.print(f"Loaded {len(chunks)} chunks from {cached}")
    else:
        with console.status("Extracting documents…"):
            docs = list(iter_documents(settings.raw_dir))
        with console.status("Chunking…"):
            chunks = chunk_documents(docs, cfg)
        console.print(f"Chunked {len(docs)} documents into {len(chunks)} chunks")

    if not chunks:
        console.print("[red]No chunks. Run `hcrag ingest` first.[/]")
        raise typer.Exit(1)

    rag_index = RagIndex.build(chunks, name=name, backend=backend, chunk_config=cfg)
    path = rag_index.save()

    console.print(
        Panel(
            f"Chunks:    [bold]{rag_index.manifest['n_chunks']}[/]\n"
            f"Documents: [bold]{rag_index.manifest['n_documents']}[/]\n"
            f"Embedder:  {rag_index.manifest['embedding_model']}\n"
            f"Backend:   {rag_index.manifest['vector_backend']}\n"
            f"Path:      {path}",
            title=f"Index '{name}' built",
        )
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Clinical question"),
    strategy: str = typer.Option("hybrid", help="vector | bm25 | hybrid | hybrid_weighted"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    top_k: int = typer.Option(20),
    top_n: int = typer.Option(5),
    index_name: str = typer.Option("default", "--index"),
) -> None:
    """Module 7 - ask a question and print the grounded answer with citations."""
    from src.generation.pipeline import RAGConfig, build_pipeline

    cfg = RAGConfig(strategy=strategy, rerank=rerank, top_k=top_k, top_n=top_n)
    with console.status("Loading index and models…"):
        pipeline = build_pipeline(index_name, cfg)
        # Otherwise the reported retrieval latency is mostly model loading.
        pipeline.warmup()

    with console.status("Retrieving and generating…"):
        result = pipeline.answer(question)

    console.print(Panel(result.answer, title="Answer", border_style="green"))

    if result.citations:
        table = Table(title="Sources", show_lines=False)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Document", max_width=48)
        table.add_column("Page", width=5)
        table.add_column("Section", max_width=30)
        for c in result.citations:
            table.add_row(str(c.marker), c.title, str(c.page), c.section)
        console.print(table)
    else:
        console.print("[yellow]No citations returned.[/]")

    t = result.timings
    console.print(
        f"[dim]strategy={result.strategy}  model={result.model}  "
        f"confidence={result.confidence:.2f}\n"
        f"retrieval={t.retrieval_latency:.2f}s  rerank={t.rerank_latency:.2f}s  "
        f"generation={t.generation_latency:.2f}s  total={t.total_latency:.2f}s  "
        f"tokens={result.usage.total_tokens}[/]"
    )


@app.command()
def evaluate(
    limit: int = typer.Option(None, help="Only evaluate the first N questions"),
    strategy: str = typer.Option("hybrid"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Run LLM-judge metrics"),
    index_name: str = typer.Option("default", "--index"),
) -> None:
    """Module 9 - evaluate retrieval and generation quality."""
    from src.evaluation.rag_evaluation import run_evaluation
    from src.generation.pipeline import RAGConfig

    cfg = RAGConfig(strategy=strategy, rerank=rerank)
    report = run_evaluation(config=cfg, limit=limit, use_judge=judge, index_name=index_name)
    console.print(report.render())


@app.command()
def compare(
    limit: int = typer.Option(None),
    judge: bool = typer.Option(True, "--judge/--no-judge"),
    index_name: str = typer.Option("default", "--index"),
    gen_arms: int = typer.Option(
        1, help="How many arms to measure generation on (0 = retrieval only)"
    ),
    gen_limit: int = typer.Option(
        25, help="Questions per arm for generation metrics; free tiers cap daily requests"
    ),
    leakage_threshold: float = typer.Option(
        0.5, help="Question/passage term overlap above which a question counts as leaky"
    ),
) -> None:
    """Module 10 - compare Vector vs BM25 vs Hybrid vs Hybrid+Reranker.

    Retrieval is scored for every arm on every question with no LLM calls.
    Generation metrics are expensive, so by default they run on the best arm
    only, chosen from the low-leakage slice.
    """
    from src.evaluation.experiments import run_strategy_comparison

    results = run_strategy_comparison(
        limit=limit,
        use_judge=judge,
        index_name=index_name,
        generation_arms=gen_arms if gen_arms > 0 else 0,
        generation_limit=gen_limit,
        leakage_threshold=leakage_threshold,
    )
    console.print(results.render())


@app.command()
def sweep(
    sizes: str = typer.Option("300,500,800,1200", help="Comma-separated chunk sizes"),
    limit: int = typer.Option(None),
) -> None:
    """Module 3 - chunk-size experiment (retrieval metrics per chunk size)."""
    from src.evaluation.experiments import run_chunk_sweep

    token_sizes = [int(s) for s in sizes.split(",")]
    results = run_chunk_sweep(token_sizes, limit=limit)
    console.print(results.render())


@app.command("make-evalset")
def make_evalset(
    n: int = typer.Option(60, help="Number of questions to generate"),
    index_name: str = typer.Option("default", "--index"),
    out: Path = typer.Option(None, help="Output path"),
) -> None:
    """Bootstrap a labelled evaluation set from the indexed corpus."""
    from src.evaluation.dataset import generate_evaluation_set

    path = generate_evaluation_set(n=n, index_name=index_name, out=out)
    console.print(f"[green]Evaluation set written to {path}[/]")


@app.command()
def info(index_name: str = typer.Option("default", "--index")) -> None:
    """Show corpus, index and LLM provider status."""
    from src.generation.llm import get_llm

    settings = get_settings()
    raw_counts = {
        p.name: len(list(p.glob("*")))
        for p in sorted(settings.raw_dir.iterdir())
        if p.is_dir()
    }

    table = Table(title="Healthcare RAG status")
    table.add_column("Component", style="cyan")
    table.add_column("Status")

    table.add_row("Raw corpus", ", ".join(f"{k}={v}" for k, v in raw_counts.items()) or "empty")

    index_path = settings.index_dir / index_name
    if (index_path / "manifest.json").exists():
        manifest = json.loads((index_path / "manifest.json").read_text())
        table.add_row(
            f"Index '{index_name}'",
            f"{manifest['n_chunks']} chunks / {manifest['n_documents']} docs "
            f"({manifest['embedding_model']}, {manifest['vector_backend']})",
        )
    else:
        table.add_row(f"Index '{index_name}'", "[yellow]not built — run `hcrag index`[/]")

    llm = get_llm()
    table.add_row("LLM provider", f"{llm.provider} ({llm.model})")
    table.add_row("Embedding model", settings.embedding_model)
    table.add_row("Reranker", settings.reranker_model)
    table.add_row("Device", settings.resolve_device())

    eval_file = settings.eval_dir / "questions.json"
    table.add_row(
        "Evaluation set",
        f"{len(json.loads(eval_file.read_text()))} questions"
        if eval_file.exists()
        else "[yellow]none — run `hcrag make-evalset`[/]",
    )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Module 11 - run the FastAPI backend."""
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


@app.command()
def ui(port: int = typer.Option(8501)) -> None:
    """Module 12 - run the Streamlit frontend."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", str(port)]
    )


if __name__ == "__main__":
    app()
