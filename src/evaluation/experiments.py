"""Module 3 + Module 10 - the experiments that produce comparable numbers.

``run_strategy_comparison``  Vector vs BM25 vs Hybrid vs Hybrid+Reranker
``run_chunk_sweep``          300 / 500 / 800 / 1200-token chunking

Both write JSON into ``data/evaluation/results/`` so the notebooks and the
Streamlit dashboard render the same figures the CLI prints.

A note on the chunk sweep: chunk IDs are a function of the chunking itself, so
gold *chunk* labels cannot survive a change in chunk size. The sweep therefore
scores at **document** granularity, which is stable across configurations and
is the only honest way to compare them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import EvalQuestion
from src.evaluation.dataset import load_evaluation_set, stratify_by_leakage
from src.evaluation.rag_evaluation import EvalReport, run_evaluation
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.generation.pipeline import RAGConfig, RAGPipeline
from src.ingestion.chunker import ChunkConfig, chunk_documents
from src.retrieval.index import RagIndex

log = get_logger(__name__)


def _results_dir() -> Path:
    path = get_settings().eval_dir / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Module 10 - strategy comparison
# ---------------------------------------------------------------------------

def _generation_sample(
    questions: list[EvalQuestion], limit: int | None
) -> list[EvalQuestion]:
    """Subsample for generation metrics, always keeping the refusal probes.

    Taking the first N questions silently dropped every unanswerable
    question, because the curated ones are appended after the generated
    ones — so refusal rate, the clearest hallucination signal in the suite,
    came back as "no data". The unanswerable questions are cheap and are the
    whole point, so they are always included.
    """
    if not limit or limit >= len(questions):
        return questions

    unanswerable = [q for q in questions if q.unanswerable]
    answerable = [q for q in questions if not q.unanswerable]
    keep_unanswerable = unanswerable[: max(limit // 4, 1)]
    remaining = max(limit - len(keep_unanswerable), 0)
    return answerable[:remaining] + keep_unanswerable


DEFAULT_ARMS: list[tuple[str, RAGConfig]] = [
    ("Vector only", RAGConfig(strategy="vector", rerank=False)),
    ("BM25 only", RAGConfig(strategy="bm25", rerank=False)),
    ("Hybrid (RRF)", RAGConfig(strategy="hybrid", rerank=False)),
    ("Hybrid + Reranker", RAGConfig(strategy="hybrid", rerank=True)),
]


@dataclass
class ComparisonResults:
    reports: dict[str, EvalReport] = field(default_factory=dict)
    # slice name -> arm -> retrieval metrics
    slices: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    slice_sizes: dict[str, int] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "arms": {name: report.to_dict() for name, report in self.reports.items()},
            "slices": self.slices,
            "slice_sizes": self.slice_sizes,
        }

    def render_slices(self) -> Table | None:
        if not self.slices:
            return None
        table = Table(
            title="Retrieval by question leakage "
            "(how much question vocabulary is copied from the gold passage)"
        )
        table.add_column("Slice", style="cyan")
        table.add_column("n", justify="right")
        table.add_column("Strategy")
        table.add_column("Recall@1", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("nDCG@5", justify="right")

        for slice_name, arms in self.slices.items():
            for i, (arm, metrics) in enumerate(arms.items()):
                table.add_row(
                    slice_name if i == 0 else "",
                    str(self.slice_sizes.get(slice_name, "")) if i == 0 else "",
                    arm,
                    f"{metrics.get('recall@1', 0):.3f}",
                    f"{metrics.get('mrr', 0):.3f}",
                    f"{metrics.get('ndcg@5', 0):.3f}",
                )
            table.add_section()
        return table

    def save(self, path: Path | None = None) -> Path:
        path = path or (_results_dir() / "strategy_comparison.json")
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def render(self) -> Group:
        table = Table(title="Retrieval strategy comparison", show_lines=False)
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column("Recall@5", justify="right")
        table.add_column("Recall@20", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("nDCG@5", justify="right")
        table.add_column("Faithful.", justify="right")
        table.add_column("Ans.Rel.", justify="right")
        table.add_column("Cite acc.", justify="right")
        table.add_column("Latency", justify="right")

        def fmt(value: float | None) -> str:
            return "—" if value is None else f"{value:.3f}"

        # Highlight the winner on the headline retrieval metric.
        best = max(
            (r.retrieval_doc.recall.get(5, 0.0) for r in self.reports.values()), default=0.0
        )
        for name, report in self.reports.items():
            recall5 = report.retrieval_doc.recall.get(5, 0.0)
            label = f"[bold green]{name}[/]" if recall5 >= best > 0 else name
            table.add_row(
                label,
                fmt(recall5),
                fmt(report.retrieval_doc.recall.get(20)),
                fmt(report.retrieval_doc.mrr),
                fmt(report.retrieval_doc.ndcg.get(5)),
                fmt(report.faithfulness),
                fmt(report.answer_relevance),
                fmt(report.citation_accuracy),
                f"{report.mean_total_latency:.2f}s",
            )

        parts = [
            Panel("Module 10 — head-to-head retrieval comparison", title="Experiment"),
            table,
        ]
        slices = self.render_slices()
        if slices is not None:
            parts.append(slices)
        return Group(*parts)


def run_strategy_comparison(
    arms: list[tuple[str, RAGConfig]] | None = None,
    limit: int | None = None,
    use_judge: bool = True,
    index_name: str = "default",
    generation_arms: int | None = 1,
    generation_limit: int | None = 25,
    leakage_threshold: float = 0.5,
) -> ComparisonResults:
    """Compare retrieval strategies, then measure generation on the best arms.

    Retrieval is scored for every arm on every question with no LLM involved,
    which is both free and fast. Generation metrics cost one LLM call per
    question plus three judge calls, so by default they are measured only on
    the best-performing arm(s) over a subsample — enough to characterise
    answer quality without exhausting a free-tier daily request budget.

    Set ``generation_arms=None`` to run generation on every arm.
    """
    arms = arms or DEFAULT_ARMS
    # Load the index and evaluation set once; every arm shares them so the
    # only thing varying between arms is the retrieval configuration.
    index = RagIndex.load(index_name)
    questions = load_evaluation_set()
    if limit:
        questions = questions[:limit]

    results = ComparisonResults()
    for name, config in arms:
        log.info("=== arm (retrieval): %s ===", name)
        results.reports[name] = run_evaluation(
            config=config,
            questions=questions,
            use_judge=False,
            use_llm=False,
            index=index,
            save=False,
        )

    # Stratify by lexical leakage. Aggregate numbers are dominated by
    # questions that reuse their source passage's vocabulary, which flatters
    # BM25 and pushes every arm to the ceiling; the low-leakage slice is
    # where the strategies actually separate.
    strata = stratify_by_leakage(questions, threshold=leakage_threshold)
    for slice_name, subset in strata.items():
        if len(subset) < 5:
            continue
        results.slice_sizes[slice_name] = len(subset)
        results.slices[slice_name] = {}
        for name, config in arms:
            sub_report = run_evaluation(
                config=config,
                questions=subset,
                use_judge=False,
                use_llm=False,
                index=index,
                save=False,
            )
            results.slices[slice_name][name] = sub_report.retrieval_doc.to_dict()

    # Pick the arms to measure generation on. Rank by MRR on the low-leakage
    # slice when it exists: ranking on the full set would select whichever arm
    # best exploits the leakage rather than the one that retrieves best.
    low = results.slices.get("low_leakage")
    if low:
        ranked = sorted(
            results.reports.items(),
            key=lambda kv: low.get(kv[0], {}).get("mrr", 0.0),
            reverse=True,
        )
    else:
        ranked = sorted(
            results.reports.items(),
            key=lambda kv: kv[1].retrieval_doc.recall.get(5, 0.0),
            reverse=True,
        )
    selected = ranked if generation_arms is None else ranked[: max(generation_arms, 0)]
    gen_questions = _generation_sample(questions, generation_limit)

    for name, _ in selected:
        config = dict(arms)[name]
        log.info("=== arm (generation): %s on %d questions ===", name, len(gen_questions))
        gen_report = run_evaluation(
            config=config,
            questions=gen_questions,
            use_judge=use_judge,
            use_llm=True,
            index=index,
            save=False,
        )
        # Merge rather than replace: the retrieval numbers were measured over
        # the full question set and must not be overwritten by the subsample.
        target = results.reports[name]
        target.model = gen_report.model
        target.n_generation_questions = len(gen_questions)
        for attr in (
            "faithfulness",
            "answer_relevance",
            "context_relevance",
            "citation_accuracy",
            "citation_coverage",
            "refusal_rate_unanswerable",
            "false_refusal_rate",
            "mean_generation_latency",
            "mean_total_latency",
            "p95_total_latency",
            "mean_tokens",
            "total_cost_usd",
            "per_question",
        ):
            setattr(target, attr, getattr(gen_report, attr))

    path = results.save()
    log.info("comparison saved to %s", path)
    return results


# ---------------------------------------------------------------------------
# Module 3 - chunk size sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepResults:
    rows: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def save(self, path: Path | None = None) -> Path:
        path = path or (_results_dir() / "chunk_sweep.json")
        path.write_text(
            json.dumps({"generated_at": self.generated_at, "rows": self.rows}, indent=2),
            encoding="utf-8",
        )
        return path

    def render(self) -> Group:
        table = Table(title="Chunk size experiment (hybrid retrieval, document-level)")
        table.add_column("Chunk tokens", style="cyan", justify="right")
        table.add_column("Overlap", justify="right")
        table.add_column("Chunks", justify="right")
        table.add_column("Mean tok", justify="right")
        table.add_column("Recall@5", justify="right")
        table.add_column("Recall@20", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("nDCG@5", justify="right")
        table.add_column("Index s", justify="right")
        table.add_column("Query ms", justify="right")

        best = max((r["recall@5"] for r in self.rows), default=0.0)
        for row in self.rows:
            label = (
                f"[bold green]{row['target_tokens']}[/]"
                if row["recall@5"] >= best > 0
                else str(row["target_tokens"])
            )
            table.add_row(
                label,
                str(row["overlap_tokens"]),
                str(row["n_chunks"]),
                f"{row['mean_tokens']:.0f}",
                f"{row['recall@5']:.3f}",
                f"{row['recall@20']:.3f}",
                f"{row['mrr']:.3f}",
                f"{row['ndcg@5']:.3f}",
                f"{row['index_seconds']:.1f}",
                f"{row['query_ms']:.1f}",
            )

        return Group(
            Panel(
                "Module 3 — does chunk size actually change retrieval quality?",
                title="Experiment",
            ),
            table,
        )


def run_chunk_sweep(
    token_sizes: list[int] | None = None,
    overlap_ratio: float = 0.15,
    limit: int | None = None,
    strategy: str = "hybrid",
    questions: list[EvalQuestion] | None = None,
) -> SweepResults:
    from src.common.schemas import Chunk
    from src.embeddings.embedder import get_embedder
    from src.ingestion.loader import iter_documents

    settings = get_settings()

    # Load the embedding model before any FAISS index exists in this process.
    # Constructing a SentenceTransformer *after* faiss has built an index
    # segfaults (both share an OpenMP runtime), and the sweep is the one code
    # path that hits that order naturally: every chunk is already cached, so
    # nothing forces the model to load until after the first index is built.
    get_embedder().encode_query("warmup")
    token_sizes = token_sizes or [300, 500, 800, 1200]
    questions = questions or load_evaluation_set()
    # Only labelled questions can produce retrieval metrics.
    questions = [q for q in questions if q.relevant_document_ids]
    if limit:
        questions = questions[:limit]

    cached = settings.processed_dir / "documents.jsonl"
    if cached.exists():
        from src.common.schemas import Document

        with cached.open(encoding="utf-8") as fh:
            docs = [Document.model_validate_json(line) for line in fh if line.strip()]
    else:
        docs = list(iter_documents(settings.raw_dir))
    log.info("chunk sweep over %d documents, %d questions", len(docs), len(questions))

    results = SweepResults()
    for target in token_sizes:
        cfg = ChunkConfig(target_tokens=target, overlap_tokens=int(target * overlap_ratio))
        log.info("=== chunk config %s ===", cfg.name)

        chunks: list[Chunk] = chunk_documents(docs, cfg)
        t0 = time.perf_counter()
        index = RagIndex.build(chunks, name=f"sweep_{cfg.name}", chunk_config=cfg)
        index_seconds = time.perf_counter() - t0

        pipeline = RAGPipeline(index, config=RAGConfig(strategy=strategy, rerank=False))

        retrieved = []
        t0 = time.perf_counter()
        for question in questions:
            contexts, _, _ = pipeline.retrieve(question.question)
            retrieved.append(contexts)
        query_ms = (time.perf_counter() - t0) / max(len(questions), 1) * 1000

        metrics = evaluate_retrieval(questions, retrieved, granularity="document")
        results.rows.append(
            {
                "target_tokens": target,
                "overlap_tokens": cfg.overlap_tokens,
                "n_chunks": len(chunks),
                "mean_tokens": sum(c.token_count for c in chunks) / max(len(chunks), 1),
                # Recall@1 discriminates where Recall@5 is at the ceiling.
                "recall@1": metrics.recall.get(1, 0.0),
                "recall@5": metrics.recall.get(5, 0.0),
                "recall@20": metrics.recall.get(20, 0.0),
                "precision@5": metrics.precision.get(5, 0.0),
                "mrr": metrics.mrr,
                "ndcg@5": metrics.ndcg.get(5, 0.0),
                "index_seconds": index_seconds,
                "query_ms": query_ms,
            }
        )
        log.info("  %s -> %s", cfg.name, metrics.summary_line())

    path = results.save()
    log.info("sweep saved to %s", path)
    return results
