"""Evaluation-set integrity: leakage measurement, filtering and stratification.

These guard the thing that made the first comparison run wrong — a benchmark
that looked excellent while measuring string overlap instead of retrieval.
"""

from __future__ import annotations

import pytest

from src.common.schemas import EvalQuestion
from src.evaluation.dataset import (
    is_self_contained,
    lexical_overlap,
    stratify_by_leakage,
)

CHUNK = (
    "Angiotensin converting enzyme inhibitors are recommended as initial "
    "pharmacological therapy for adults with hypertension and diabetes. "
    "Monitor serum potassium and creatinine within two weeks of initiation."
)


class TestLexicalOverlap:
    def test_verbatim_question_scores_one(self):
        """The failure mode that produced BM25's perfect Recall@5: every
        content word in the question also appears in its source passage."""
        question = "Which therapy is recommended as initial pharmacological therapy?"
        assert lexical_overlap(question, CHUNK) == pytest.approx(1.0)

    def test_paraphrase_scores_low(self):
        question = "Which blood pressure drugs should a clinician start first?"
        assert lexical_overlap(question, CHUNK) < 0.4

    def test_unrelated_question_scores_zero(self):
        assert lexical_overlap("How is asthma diagnosed in children?", CHUNK) == 0.0

    def test_empty_question_is_safe(self):
        assert lexical_overlap("", CHUNK) == 0.0

    def test_partial_overlap_is_proportional(self):
        score = lexical_overlap("What is the recommended dose of ibuprofen?", CHUNK)
        assert 0.0 < score < 1.0


class TestSelfContainment:
    @pytest.mark.parametrize(
        "question",
        [
            "What did this study find about mortality?",
            "Which drugs were used in the trial?",
            "What are the limitations of this systematic review?",
            "What did this systematic scoping review conclude?",
            "Which of the following is recommended?",
            "How does the above therapy work?",
        ],
    )
    def test_rejects_anaphoric_questions(self, question):
        """A retriever cannot resolve "this study", so such a question
        measures nothing and must never enter the test set."""
        assert not is_self_contained(question)

    @pytest.mark.parametrize(
        "question",
        [
            "What are the first-line treatments for hypertension?",
            "Which adverse reactions are associated with amlodipine?",
            "How do SGLT2 inhibitors lower blood glucose?",
        ],
    )
    def test_accepts_standalone_questions(self, question):
        assert is_self_contained(question)

    def test_rejects_fragments(self):
        assert not is_self_contained("Hypertension?")

    def test_therapy_is_not_treated_as_anaphora(self):
        """'the treatment of X' is a normal clinical phrase, not a reference."""
        assert is_self_contained("What is the treatment of choice for sepsis?")


class TestGenerationSample:
    """Refusal rate is the clearest hallucination signal, and taking the
    first N questions dropped every unanswerable probe."""

    def _questions(self, n_answerable: int = 40, n_unanswerable: int = 5):
        from src.evaluation.experiments import _generation_sample  # noqa: F401

        answerable = [
            EvalQuestion(question_id=f"a{i}", question="q", expected_information="")
            for i in range(n_answerable)
        ]
        unanswerable = [
            EvalQuestion(
                question_id=f"u{i}", question="q", expected_information="", unanswerable=True
            )
            for i in range(n_unanswerable)
        ]
        return answerable + unanswerable

    def test_sample_always_keeps_refusal_probes(self):
        from src.evaluation.experiments import _generation_sample

        sample = _generation_sample(self._questions(), limit=25)
        assert len(sample) == 25
        assert any(q.unanswerable for q in sample)

    def test_sample_respects_the_limit(self):
        from src.evaluation.experiments import _generation_sample

        assert len(_generation_sample(self._questions(), limit=8)) == 8

    def test_no_limit_returns_everything(self):
        from src.evaluation.experiments import _generation_sample

        questions = self._questions()
        assert _generation_sample(questions, limit=None) == questions

    def test_handles_sets_without_unanswerables(self):
        from src.evaluation.experiments import _generation_sample

        sample = _generation_sample(self._questions(n_unanswerable=0), limit=10)
        assert len(sample) == 10


class TestStratification:
    def _questions(self) -> list[EvalQuestion]:
        return [
            EvalQuestion(
                question_id=f"q{i}",
                question="q",
                expected_information="",
                relevant_document_ids=["d1"],
                lexical_overlap=overlap,
            )
            for i, overlap in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
        ]

    def test_splits_on_threshold_inclusively(self):
        strata = stratify_by_leakage(self._questions(), threshold=0.5)
        assert len(strata["low_leakage"]) == 3  # 0.1, 0.3, 0.5
        assert len(strata["high_leakage"]) == 2  # 0.7, 0.9

    def test_slices_partition_the_labelled_set(self):
        strata = stratify_by_leakage(self._questions(), threshold=0.5)
        assert len(strata["low_leakage"]) + len(strata["high_leakage"]) == len(strata["all"])

    def test_unlabelled_questions_are_excluded(self):
        questions = self._questions() + [
            EvalQuestion(question_id="u", question="q", expected_information="")
        ]
        assert len(stratify_by_leakage(questions)["all"]) == 5
