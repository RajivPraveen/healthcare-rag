"""Citation extraction, confidence, grounding and the offline LLM fallback."""

from __future__ import annotations

import pytest

from src.common.schemas import Answer, Chunk, ScoredChunk
from src.evaluation.rag_evaluation import citation_accuracy, citation_coverage, is_refusal
from src.generation.llm import ExtractiveClient, RateLimiter
from src.generation.pipeline import (
    compute_confidence,
    extract_citations,
    normalize_citation_markers,
)
from src.generation.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt, format_context


def _context(idx: int, text: str = "Clinical evidence text.") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"c{idx}",
        document_id=f"d{idx}",
        title=f"Guideline {idx}",
        source=f"g{idx}.pdf",
        page=idx + 1,
        section="Treatment",
        text=text,
        token_count=10,
    )
    return ScoredChunk(chunk=chunk, score=0.8, rerank_score=2.0)


CONTEXTS = [_context(i) for i in range(3)]


class TestCitationExtraction:
    def test_maps_markers_to_passages(self):
        citations, markers = extract_citations("ACE inhibitors help [1]. Also [3].", CONTEXTS)
        assert markers == {1, 3}
        assert [c.marker for c in citations] == [1, 3]
        assert citations[0].page == 1 and citations[1].page == 3

    def test_drops_out_of_range_markers(self):
        """A citation pointing at a passage that was never retrieved is worse
        than no citation, so it must not reach the UI."""
        citations, markers = extract_citations("Claim [9].", CONTEXTS)
        assert citations == [] and markers == set()

    def test_no_markers_yields_no_citations(self):
        citations, markers = extract_citations("An uncited assertion.", CONTEXTS)
        assert citations == [] and markers == set()

    def test_citation_carries_snippet_and_provenance(self):
        citation = extract_citations("Text [2].", CONTEXTS)[0][0]
        assert citation.document_id == "d1"
        assert citation.section == "Treatment"
        assert citation.snippet


class TestCitationNormalization:
    """gpt-oss emits OpenAI file-citation tokens rather than plain brackets.

    Left unhandled this parsed as zero citations and reported a fully
    grounded answer as ungrounded, so each variant has a regression test.
    """

    def test_cjk_file_citation_token(self):
        assert normalize_citation_markers("Claim 【1†L31-L33】.") == "Claim [1]."

    def test_bare_cjk_marker(self):
        assert normalize_citation_markers("Claim 【2】.") == "Claim [2]."

    def test_grouped_markers_are_split(self):
        assert normalize_citation_markers("Claim [1, 3].") == "Claim [1][3]."

    def test_source_word_prefix(self):
        assert normalize_citation_markers("Claim [Source 2].") == "Claim [2]."

    def test_canonical_form_untouched(self):
        assert normalize_citation_markers("Claim [1][2].") == "Claim [1][2]."

    def test_normalized_markers_become_citations(self):
        citations, markers = extract_citations(
            normalize_citation_markers("A 【1†L1-L2】. B 【3†L9】."), CONTEXTS
        )
        assert markers == {1, 3}
        assert len(citations) == 2


class TestConfidence:
    def test_refusal_scores_zero(self):
        assert compute_confidence("No info.", CONTEXTS, set(), refused=True) == 0.0

    def test_empty_context_scores_zero(self):
        assert compute_confidence("Anything.", [], set(), refused=False) == 0.0

    def test_more_citations_raises_confidence(self):
        answer = "Claim one [1]. Claim two [2]. Claim three [3]."
        low = compute_confidence("Claim one [1].", CONTEXTS, {1}, refused=False)
        high = compute_confidence(answer, CONTEXTS, {1, 2, 3}, refused=False)
        assert high > low

    def test_uncited_answer_scores_lower(self):
        cited = compute_confidence("A claim about therapy [1].", CONTEXTS, {1}, refused=False)
        uncited = compute_confidence("A claim about therapy.", CONTEXTS, set(), refused=False)
        assert cited > uncited

    def test_bounded_to_unit_interval(self):
        score = compute_confidence("A [1]. B [2]. C [3].", CONTEXTS, {1, 2, 3}, refused=False)
        assert 0.0 <= score <= 1.0


class TestPrompts:
    def test_system_prompt_states_the_safety_rules(self):
        lowered = ANSWER_SYSTEM_PROMPT.lower()
        assert "only" in lowered
        assert "do not contain" in lowered or "no answer" in lowered
        assert "cit" in lowered

    def test_context_is_numbered_with_provenance(self):
        rendered = format_context(CONTEXTS)
        assert "[1]" in rendered and "[3]" in rendered
        assert "page" in rendered and "section" in rendered

    def test_empty_context_forces_refusal(self):
        assert "do not contain information" in build_answer_prompt("Q?", [])


class TestExtractiveFallback:
    def test_answers_from_context_without_a_model(self):
        prompt = build_answer_prompt("What is recommended?", CONTEXTS)
        response = ExtractiveClient().complete(ANSWER_SYSTEM_PROMPT, prompt)
        assert response.text
        assert response.provider == "extractive"

    def test_refuses_when_nothing_retrieved(self):
        response = ExtractiveClient().complete(ANSWER_SYSTEM_PROMPT, build_answer_prompt("Q?", []))
        assert "could not be found" in response.text.lower()

    def test_reports_zero_cost(self):
        assert ExtractiveClient().complete("s", "u").cost_usd == 0.0


class TestEvalHelpers:
    def _answer(self, text: str) -> Answer:
        return Answer(question="q", answer=text, contexts=CONTEXTS)

    def test_citation_accuracy_counts_valid_markers(self):
        assert citation_accuracy(self._answer("A [1]. B [2].")) == 1.0
        assert citation_accuracy(self._answer("A [1]. B [9].")) == 0.5

    def test_citation_accuracy_is_none_without_citations(self):
        """Uncited refusals must not be averaged in as zeros."""
        assert citation_accuracy(self._answer("No citations here.")) is None

    def test_citation_coverage_measures_sentences(self):
        answer = self._answer(
            "ACE inhibitors are recommended as first-line therapy [1]. "
            "Thiazide diuretics are an equally valid initial choice."
        )
        assert citation_coverage(answer) == 0.5

    def test_citation_coverage_ignores_short_fragments(self):
        """Sentences under four words are headers or fragments, not claims,
        and must not dilute the coverage denominator."""
        answer = self._answer("Sources below. ACE inhibitors are first-line therapy [1].")
        assert citation_coverage(answer) == 1.0

    def test_detects_refusal(self):
        assert is_refusal(
            self._answer("The provided documents do not contain information to answer this.")
        )
        assert not is_refusal(self._answer("ACE inhibitors are first-line [1]."))


class TestRateLimiter:
    def test_disabled_limiter_never_blocks(self):
        RateLimiter(0, 0).acquire(10_000)  # must return immediately

    def test_token_window_admits_within_budget(self):
        limiter = RateLimiter(requests_per_minute=0, tokens_per_minute=1000)
        limiter.acquire(400)
        limiter.acquire(400)
        assert sum(t for _, t in limiter._events) == 800

    def test_actual_usage_corrects_the_window(self):
        limiter = RateLimiter(tokens_per_minute=1000)
        limiter.acquire(100)
        limiter.record_actual(100, 250)
        assert limiter._events[-1][1] == 250


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer [1].", {1}),
        ("Answer [1][2].", {1, 2}),
        ("Answer [1] and [2], plus [3].", {1, 2, 3}),
        ("No markers.", set()),
    ],
)
def test_marker_parsing_variants(text, expected):
    assert extract_citations(text, CONTEXTS)[1] == expected
