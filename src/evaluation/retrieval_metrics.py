"""Module 9 (part 1) - retrieval metrics.

Standard IR measures, computed against a labelled set where each question
carries the chunk(s) and document(s) that genuinely answer it.

Which metric matters depends on the stage:

* **Recall@K** is the ceiling on the whole system. Anything the retriever
  misses at K, the reranker and the LLM can never recover. This is the number
  to optimise for the first stage.
* **Precision@K** matters at the *reranked* K, because those passages become
  prompt tokens. Low precision means paying to feed the model noise.
* **MRR** captures how near the top the first correct passage lands. LLMs
  attend unevenly across a long context, so rank position has real effect.
* **nDCG@K** is the rank-sensitive quality measure when several passages are
  relevant.
* **Hit Rate@K** is the blunt "did we get anything useful at all" check.

Relevance is evaluated at both chunk and document granularity: chunk-level is
strict (did we find *the* passage), document-level is fairer, since a fact
often appears in several chunks of the same paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean

from src.common.schemas import EvalQuestion, ScoredChunk

DEFAULT_KS = (1, 3, 5, 10, 20)


# ---------------------------------------------------------------------------
# Single-query metrics
# ---------------------------------------------------------------------------


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return len([r for r in top if r in relevant]) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), len(retrieved)) if hits else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG: DCG normalised by the best achievable ordering."""
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """Averages over an evaluation set, at one relevance granularity."""

    granularity: str = "document"
    n_questions: int = 0
    hit_rate: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "granularity": self.granularity,
            "n_questions": self.n_questions,
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            **{f"hit_rate@{k}": round(v, 4) for k, v in self.hit_rate.items()},
            **{f"precision@{k}": round(v, 4) for k, v in self.precision.items()},
            **{f"recall@{k}": round(v, 4) for k, v in self.recall.items()},
            **{f"ndcg@{k}": round(v, 4) for k, v in self.ndcg.items()},
        }

    def summary_line(self) -> str:
        return (
            f"Recall@5={self.recall.get(5, 0):.3f}  "
            f"Recall@20={self.recall.get(20, 0):.3f}  "
            f"MRR={self.mrr:.3f}  "
            f"nDCG@5={self.ndcg.get(5, 0):.3f}"
        )


def _ids(hits: list[ScoredChunk], granularity: str) -> list[str]:
    if granularity == "chunk":
        return [h.chunk.chunk_id for h in hits]
    return _dedupe([h.chunk.document_id for h in hits])


def _dedupe(items: list[str]) -> list[str]:
    """Collapse repeats while preserving order.

    Document-level ranks must be computed on *distinct* documents, otherwise
    three chunks from one paper would inflate precision and distort MRR.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _relevant(question: EvalQuestion, granularity: str) -> set[str]:
    if granularity == "chunk":
        return set(question.relevant_chunk_ids)
    return set(question.relevant_document_ids)


def evaluate_retrieval(
    questions: list[EvalQuestion],
    retrieved_per_question: list[list[ScoredChunk]],
    ks: tuple[int, ...] = DEFAULT_KS,
    granularity: str = "document",
) -> RetrievalMetrics:
    """Average the per-query metrics across an evaluation set."""
    per_k_hit: dict[int, list[float]] = {k: [] for k in ks}
    per_k_prec: dict[int, list[float]] = {k: [] for k in ks}
    per_k_rec: dict[int, list[float]] = {k: [] for k in ks}
    per_k_ndcg: dict[int, list[float]] = {k: [] for k in ks}
    rrs: list[float] = []
    aps: list[float] = []

    counted = 0
    for question, hits in zip(questions, retrieved_per_question):
        relevant = _relevant(question, granularity)
        if not relevant:
            # Unlabelled questions would otherwise drag every metric to zero.
            continue
        counted += 1
        retrieved = _ids(hits, granularity)

        for k in ks:
            per_k_hit[k].append(hit_rate_at_k(retrieved, relevant, k))
            per_k_prec[k].append(precision_at_k(retrieved, relevant, k))
            per_k_rec[k].append(recall_at_k(retrieved, relevant, k))
            per_k_ndcg[k].append(ndcg_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
        aps.append(average_precision(retrieved, relevant))

    def avg(values: list[float]) -> float:
        return mean(values) if values else 0.0

    return RetrievalMetrics(
        granularity=granularity,
        n_questions=counted,
        hit_rate={k: avg(v) for k, v in per_k_hit.items()},
        precision={k: avg(v) for k, v in per_k_prec.items()},
        recall={k: avg(v) for k, v in per_k_rec.items()},
        ndcg={k: avg(v) for k, v in per_k_ndcg.items()},
        mrr=avg(rrs),
        map_score=avg(aps),
    )
