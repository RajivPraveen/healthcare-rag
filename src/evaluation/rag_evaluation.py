"""Module 9 (part 3) - end-to-end RAG evaluation.

Three families of metric, because a RAG system can fail in three unrelated ways:

**Retrieval** (Recall@K, MRR, nDCG) - did we even find the evidence? Computed
from the labelled evaluation set in ``retrieval_metrics``.

**Generation** (faithfulness, answer relevance, context relevance, citation
accuracy) - given the evidence, did the model use it honestly? Scored by an
LLM judge, except citation accuracy which is checked deterministically.

**Engineering** (latency per stage, tokens, cost) - measured on every query,
because a system that is accurate and takes nine seconds is not shippable.

Two of these are worth flagging as genuinely programmatic rather than
judged:

* *Citation accuracy* verifies that each marker the model emitted points at a
  passage that actually exists and was actually retrieved. No LLM needed.
* *Refusal rate* on the deliberately unanswerable questions is the single
  clearest hallucination signal in the whole suite.
"""

from __future__ import annotations

import json
import re
import statistics
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
from src.common.schemas import Answer, EvalQuestion
from src.evaluation.dataset import load_evaluation_set
from src.evaluation.retrieval_metrics import RetrievalMetrics, evaluate_retrieval
from src.generation.llm import LLMClient, get_judge_llm
from src.generation.pipeline import RAGConfig, RAGPipeline
from src.generation.prompts import (
    ANSWER_RELEVANCE_SYSTEM,
    CONTEXT_RELEVANCE_SYSTEM,
    FAITHFULNESS_SYSTEM,
    build_answer_relevance_prompt,
    build_context_relevance_prompt,
    build_faithfulness_prompt,
)
from src.retrieval.index import RagIndex

log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def _judge(llm: LLMClient, system: str, user: str) -> dict[str, Any]:
    try:
        response = llm.complete(system, user, temperature=0.0, max_tokens=400)
    except Exception as exc:
        log.warning("judge call failed: %s", exc)
        return {}
    match = _JSON_RE.search(response.text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------


def citation_accuracy(answer: Answer) -> float | None:
    """Fraction of emitted markers that point at a real retrieved passage.

    ``None`` when the answer cited nothing, so that uncited answers are
    excluded from the average rather than scored as zero — refusals legitimately
    carry no citations.
    """
    markers = [int(m) for m in _CITATION_RE.findall(answer.answer)]
    if not markers:
        return None
    valid = sum(1 for m in markers if 1 <= m <= len(answer.contexts))
    return valid / len(markers)


def citation_coverage(answer: Answer) -> float:
    """Fraction of substantive sentences carrying at least one citation."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.answer) if len(s.split()) >= 4]
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if _CITATION_RE.search(s)) / len(sentences)


def is_refusal(answer: Answer) -> bool:
    text = answer.answer.lower()
    return any(
        phrase in text
        for phrase in (
            "do not contain information",
            "does not contain information",
            "could not be found",
            "no information",
            "not contain",
        )
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    label: str = "hybrid+rerank"
    n_questions: int = 0
    # Generation metrics may be measured on a subsample of n_questions.
    n_generation_questions: int = 0
    model: str = ""
    retrieval_doc: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    retrieval_chunk: RetrievalMetrics = field(default_factory=RetrievalMetrics)

    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_relevance: float | None = None
    citation_accuracy: float = 0.0
    citation_coverage: float = 0.0
    refusal_rate_unanswerable: float | None = None
    false_refusal_rate: float | None = None

    mean_retrieval_latency: float = 0.0
    mean_rerank_latency: float = 0.0
    mean_generation_latency: float = 0.0
    mean_total_latency: float = 0.0
    p95_total_latency: float = 0.0
    mean_tokens: float = 0.0
    total_cost_usd: float = 0.0

    per_question: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "n_questions": self.n_questions,
            "n_generation_questions": self.n_generation_questions,
            "retrieval_document": self.retrieval_doc.to_dict(),
            "retrieval_chunk": self.retrieval_chunk.to_dict(),
            "generation": {
                "faithfulness": self.faithfulness,
                "answer_relevance": self.answer_relevance,
                "context_relevance": self.context_relevance,
                "citation_accuracy": round(self.citation_accuracy, 4),
                "citation_coverage": round(self.citation_coverage, 4),
                "refusal_rate_unanswerable": self.refusal_rate_unanswerable,
                "false_refusal_rate": self.false_refusal_rate,
            },
            "engineering": {
                "mean_retrieval_latency": round(self.mean_retrieval_latency, 4),
                "mean_rerank_latency": round(self.mean_rerank_latency, 4),
                "mean_generation_latency": round(self.mean_generation_latency, 4),
                "mean_total_latency": round(self.mean_total_latency, 4),
                "p95_total_latency": round(self.p95_total_latency, 4),
                "mean_tokens": round(self.mean_tokens, 1),
                "total_cost_usd": round(self.total_cost_usd, 6),
            },
        }

    def save(self, path: Path | None = None) -> Path:
        settings = get_settings()
        results_dir = settings.eval_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = path or (results_dir / f"{self.label.replace('+', '_')}_{stamp}.json")
        payload = self.to_dict()
        payload["per_question"] = self.per_question
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def render(self) -> Group:
        def bar(value: float | None, width: int = 24) -> str:
            if value is None:
                return "[dim]n/a[/]"
            filled = int(round(value * width))
            return f"[green]{'█' * filled}[/][dim]{'░' * (width - filled)}[/] {value * 100:5.1f}%"

        retrieval = Table(title="Retrieval (document-level)", show_header=True)
        retrieval.add_column("Metric", style="cyan")
        retrieval.add_column("Score", justify="left")
        for k in (1, 3, 5, 10, 20):
            retrieval.add_row(f"Recall@{k}", bar(self.retrieval_doc.recall.get(k)))
        retrieval.add_row("Precision@5", bar(self.retrieval_doc.precision.get(5)))
        retrieval.add_row("MRR", bar(self.retrieval_doc.mrr))
        retrieval.add_row("nDCG@5", bar(self.retrieval_doc.ndcg.get(5)))
        retrieval.add_row("Hit Rate@5", bar(self.retrieval_doc.hit_rate.get(5)))

        generation = Table(title="Generation", show_header=True)
        generation.add_column("Metric", style="cyan")
        generation.add_column("Score", justify="left")
        generation.add_row("Faithfulness", bar(self.faithfulness))
        generation.add_row("Answer relevance", bar(self.answer_relevance))
        generation.add_row("Context relevance", bar(self.context_relevance))
        generation.add_row("Citation accuracy", bar(self.citation_accuracy))
        generation.add_row("Citation coverage", bar(self.citation_coverage))
        generation.add_row("Refusal (unanswerable)", bar(self.refusal_rate_unanswerable))
        generation.add_row("False refusal", bar(self.false_refusal_rate))

        engineering = Table(title="Engineering", show_header=True)
        engineering.add_column("Metric", style="cyan")
        engineering.add_column("Value", justify="right")
        engineering.add_row("Retrieval latency", f"{self.mean_retrieval_latency * 1000:.0f} ms")
        engineering.add_row("Rerank latency", f"{self.mean_rerank_latency * 1000:.0f} ms")
        engineering.add_row("Generation latency", f"{self.mean_generation_latency:.2f} s")
        engineering.add_row("Total (mean)", f"{self.mean_total_latency:.2f} s")
        engineering.add_row("Total (p95)", f"{self.p95_total_latency:.2f} s")
        engineering.add_row("Tokens / query", f"{self.mean_tokens:.0f}")
        engineering.add_row("Total cost", f"${self.total_cost_usd:.4f}")

        return Group(
            Panel(
                f"[bold]{self.label}[/]   {self.n_questions} questions   model={self.model}",
                title="RAG evaluation",
            ),
            retrieval,
            generation,
            engineering,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_evaluation(
    config: RAGConfig | None = None,
    questions: list[EvalQuestion] | None = None,
    limit: int | None = None,
    use_judge: bool = True,
    use_llm: bool = True,
    index_name: str = "default",
    index: RagIndex | None = None,
    save: bool = True,
) -> EvalReport:
    """Evaluate one pipeline configuration.

    ``use_llm=False`` scores retrieval only. Comparing retrieval strategies
    does not require generating anything, and skipping the LLM turns a
    rate-limited run of many minutes into a few seconds of local compute —
    which is what makes it practical to compare every arm on every question
    rather than on a subsample.
    """
    config = config or RAGConfig()
    index = index or RagIndex.load(index_name)
    questions = questions or load_evaluation_set()
    if limit:
        questions = questions[:limit]

    pipeline = RAGPipeline(index, config=config)
    # Warm first so model loading is not charged to question #1's latency.
    pipeline.warmup()
    judge = get_judge_llm() if (use_judge and use_llm) else None
    if judge is not None and judge.provider == "extractive":
        log.warning("no judge model available; skipping LLM-judged metrics")
        judge = None

    report = EvalReport(
        label=config.label,
        n_questions=len(questions),
        model=pipeline.llm.model if use_llm else "retrieval-only",
    )

    retrieved_per_question = []
    faithfulness: list[float] = []
    answer_relevance: list[float] = []
    context_relevance: list[float] = []
    cite_acc: list[float] = []
    cite_cov: list[float] = []
    refusals_unanswerable: list[float] = []
    false_refusals: list[float] = []
    latencies: list[float] = []

    log.info("evaluating %d questions with strategy '%s'", len(questions), config.label)
    started = time.perf_counter()

    for i, question in enumerate(questions, start=1):
        if not use_llm:
            contexts, retrieval_latency, rerank_latency = pipeline.retrieve(question.question)
            retrieved_per_question.append(contexts)
            report.mean_retrieval_latency += retrieval_latency
            report.mean_rerank_latency += rerank_latency
            latencies.append(retrieval_latency + rerank_latency)
            report.per_question.append(
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "top_documents": [c.chunk.document_id for c in contexts],
                    "relevant_documents": question.relevant_document_ids,
                    "latency": round(retrieval_latency + rerank_latency, 4),
                }
            )
            if i % 25 == 0:
                log.info("  %d/%d retrieved", i, len(questions))
            continue

        result = pipeline.answer(question.question)
        retrieved_per_question.append(result.contexts)

        refused = is_refusal(result)
        if question.unanswerable:
            refusals_unanswerable.append(1.0 if refused else 0.0)
        else:
            false_refusals.append(1.0 if refused else 0.0)

        acc = citation_accuracy(result)
        if acc is not None:
            cite_acc.append(acc)
        if not question.unanswerable:
            cite_cov.append(citation_coverage(result))

        latencies.append(result.timings.total_latency)
        report.total_cost_usd += result.usage.estimated_cost_usd
        report.mean_tokens += result.usage.total_tokens
        report.mean_retrieval_latency += result.timings.retrieval_latency
        report.mean_rerank_latency += result.timings.rerank_latency
        report.mean_generation_latency += result.timings.generation_latency

        row: dict[str, Any] = {
            "question_id": question.question_id,
            "question": question.question,
            "answer": result.answer,
            "confidence": result.confidence,
            "refused": refused,
            "unanswerable": question.unanswerable,
            "n_citations": len(result.citations),
            "citation_accuracy": acc,
            "top_documents": [c.chunk.document_id for c in result.contexts],
            "relevant_documents": question.relevant_document_ids,
            "latency": result.timings.total_latency,
        }

        if judge is not None and result.contexts:
            f = _judge(
                judge, FAITHFULNESS_SYSTEM, build_faithfulness_prompt(result.answer, result.contexts)
            )
            if "score" in f:
                faithfulness.append(float(f["score"]))
                row["faithfulness"] = f["score"]
                row["unsupported_claims"] = f.get("unsupported_claims", [])

            a = _judge(
                judge,
                ANSWER_RELEVANCE_SYSTEM,
                build_answer_relevance_prompt(question.question, result.answer),
            )
            if "score" in a:
                answer_relevance.append(float(a["score"]))
                row["answer_relevance"] = a["score"]

            c = _judge(
                judge,
                CONTEXT_RELEVANCE_SYSTEM,
                build_context_relevance_prompt(question.question, result.contexts),
            )
            if "score" in c:
                context_relevance.append(float(c["score"]))
                row["context_relevance"] = c["score"]

        report.per_question.append(row)

        if i % 10 == 0:
            log.info("  %d/%d evaluated (%.1fs elapsed)", i, len(questions), time.perf_counter() - started)

    n = max(len(questions), 1)
    report.mean_tokens /= n
    report.mean_retrieval_latency /= n
    report.mean_rerank_latency /= n
    report.mean_generation_latency /= n
    report.mean_total_latency = statistics.mean(latencies) if latencies else 0.0
    report.p95_total_latency = (
        sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    )

    report.retrieval_doc = evaluate_retrieval(
        questions, retrieved_per_question, granularity="document"
    )
    report.retrieval_chunk = evaluate_retrieval(
        questions, retrieved_per_question, granularity="chunk"
    )

    def avg(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None

    report.faithfulness = avg(faithfulness)
    report.answer_relevance = avg(answer_relevance)
    report.context_relevance = avg(context_relevance)
    report.citation_accuracy = statistics.mean(cite_acc) if cite_acc else 0.0
    report.citation_coverage = statistics.mean(cite_cov) if cite_cov else 0.0
    report.refusal_rate_unanswerable = avg(refusals_unanswerable)
    report.false_refusal_rate = avg(false_refusals)

    if save:
        path = report.save()
        log.info("report saved to %s", path)

    return report
