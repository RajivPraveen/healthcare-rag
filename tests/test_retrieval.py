"""Retrieval behaviour: BM25 scoring, fusion, reranking and IR metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.schemas import Chunk, EvalQuestion, ScoredChunk
from src.evaluation.retrieval_metrics import (
    average_precision,
    evaluate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.retrieval.bm25 import BM25Index
from src.retrieval.query_processing import expand_query
from src.retrieval.reranker import Reranker
from src.retrieval.vector_search import FaissStore


def _chunk(idx: int, text: str, document_id: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"c{idx}",
        document_id=document_id or f"d{idx}",
        title=f"Doc {idx}",
        source=f"d{idx}.pdf",
        page=1,
        section="Body",
        text=text,
        token_count=len(text.split()),
    )


CORPUS = [
    _chunk(0, "ACE inhibitors and thiazide diuretics are first-line treatments for hypertension."),
    _chunk(1, "Metformin is the initial pharmacological therapy for type 2 diabetes mellitus."),
    _chunk(2, "Inhaled corticosteroids provide long-term control of persistent asthma."),
    # A distractor dense in query words but about a different condition entirely.
    _chunk(
        3,
        "The recommended first line first line treatment algorithm lists first line options "
        "and recommended first line steps for depressive disorder management.",
    ),
]


class TestBM25:
    @pytest.fixture
    def index(self) -> BM25Index:
        bm25 = BM25Index()
        bm25.build(CORPUS)
        return bm25

    def test_finds_exact_term(self, index):
        assert index.search("metformin", top_k=1)[0][0] == "c1"

    def test_unknown_term_returns_nothing(self, index):
        assert index.search("zolpidextrin", top_k=5) == []

    def test_coordination_beats_keyword_stuffing(self, index):
        """The distractor repeats query words but omits the topic term.

        Without a coordination factor, plain BM25 ranks it first. This is the
        regression test for that fix.
        """
        top_id = index.search("first-line treatments for hypertension", top_k=1)[0][0]
        assert top_id == "c0"

    def test_idf_rewards_rare_terms(self, index):
        assert index.idf["metformin"] > index.idf["first"]

    def test_roundtrip_persistence(self, index, tmp_path):
        index.save(tmp_path)
        restored = BM25Index()
        restored.load(tmp_path)
        assert restored.size == index.size
        assert restored.search("asthma", top_k=1) == index.search("asthma", top_k=1)


class TestFaissStore:
    def test_exact_search_recovers_identical_vector(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(4, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        store = FaissStore()
        store.build(CORPUS, vectors)
        assert store.size == 4

        chunk_id, score = store.search(vectors[2], top_k=1)[0]
        assert chunk_id == "c2"
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            FaissStore().build(CORPUS, np.zeros((2, 16), dtype=np.float32))

    def test_search_after_torch_inference(self):
        """Regression: faiss and torch share an OpenMP runtime, and a parallel
        faiss search after torch has run used to segfault the process."""
        from src.embeddings.embedder import get_embedder

        get_embedder().encode_query("warm the torch runtime")

        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(4, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        store = FaissStore()
        store.build(CORPUS, vectors)

        assert store.search(vectors[0], top_k=2)[0][0] == "c0"

    def test_faiss_is_pinned_to_one_thread(self):
        from src.retrieval.vector_search import _import_faiss

        assert _import_faiss().omp_get_max_threads() == 1


class TestQueryProcessing:
    def test_expands_known_abbreviation(self):
        assert "myocardial infarction" in expand_query("treatment after MI")

    def test_keeps_original_token(self):
        assert "HTN" in expand_query("HTN management")

    def test_leaves_unknown_terms_alone(self):
        assert expand_query("sepsis bundle") == "sepsis bundle"


class TestRerankerDiversity:
    def test_no_op_reranker_preserves_order(self):
        candidates = [ScoredChunk(chunk=c, score=1.0) for c in CORPUS]
        assert [c.chunk.chunk_id for c in Reranker().rerank("q", candidates, 2)] == ["c0", "c1"]

    def test_per_document_quota_diversifies(self):
        from src.retrieval.reranker import CrossEncoderReranker

        # Four chunks, all from the same document, plus one from another.
        same = [
            ScoredChunk(chunk=_chunk(i, f"text {i}", document_id="shared"), rerank_score=10 - i)
            for i in range(4)
        ]
        other = ScoredChunk(chunk=_chunk(9, "other", document_id="other"), rerank_score=0.1)

        reranker = CrossEncoderReranker(max_per_document=2)
        selected = reranker._diversify(same + [other], n=3)
        doc_ids = [s.chunk.document_id for s in selected]
        assert doc_ids.count("shared") == 2
        assert "other" in doc_ids


class TestRetrievalMetrics:
    def test_precision_and_recall(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c", "z"}
        assert precision_at_k(retrieved, relevant, 4) == 0.5
        assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)

    def test_reciprocal_rank_uses_first_hit(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
        assert reciprocal_rank(["x"], {"a"}) == 0.0

    def test_ndcg_rewards_higher_placement(self):
        top = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        bottom = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert top == 1.0
        assert top > bottom

    def test_average_precision(self):
        assert average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(
            (1 / 1 + 2 / 3) / 2
        )

    def test_document_granularity_deduplicates(self):
        """Three chunks of one document must count as a single ranked document."""
        hits = [
            ScoredChunk(chunk=_chunk(i, "t", document_id="dup")) for i in range(3)
        ] + [ScoredChunk(chunk=_chunk(9, "t", document_id="gold"))]
        question = EvalQuestion(
            question_id="q1",
            question="q",
            expected_information="",
            relevant_document_ids=["gold"],
        )
        metrics = evaluate_retrieval([question], [hits], ks=(2,), granularity="document")
        # "gold" is the 2nd distinct document, so RR = 1/2 rather than 1/4.
        assert metrics.mrr == pytest.approx(0.5)

    def test_unlabelled_questions_are_excluded(self):
        question = EvalQuestion(question_id="q", question="q", expected_information="")
        metrics = evaluate_retrieval([question], [[]], granularity="document")
        assert metrics.n_questions == 0
