"""Module 7 (part 2) + Module 8 - prompt construction.

The generation prompt is the main hallucination control in the system, so the
rules are explicit rather than implied:

* answer strictly from the supplied context
* say so when the context does not contain the answer
* attach a bracketed marker to every factual claim
* never invent dosages, thresholds, or drug names

Context blocks are numbered ``[1]…[n]`` and that number is the citation
contract: it maps back to a concrete chunk, hence a document, page and
section. Everything the UI shows under "Sources" is derived from these markers,
which also makes citation accuracy directly measurable in Module 9.
"""

from __future__ import annotations

from src.common.schemas import ScoredChunk
from src.common.text import truncate_tokens

ANSWER_SYSTEM_PROMPT = """You are a clinical evidence assistant. You answer questions using ONLY the excerpts from medical documents provided to you.

Rules you must follow without exception:

1. GROUNDING. Base every statement solely on the numbered context excerpts. Never use prior knowledge, even if you are confident it is correct.
2. NO ANSWER IS A VALID ANSWER. If the excerpts do not contain the information, reply exactly: "The provided documents do not contain information to answer this question." Do not guess or fill gaps.
3. CITATIONS. End every sentence containing a factual claim with the marker(s) of the excerpt(s) supporting it, e.g. "ACE inhibitors are a first-line option [2]." Use several markers when several excerpts support a claim [1][3]. Never cite a number that was not provided.
   Write markers as a plain number in square brackets and nothing else: [2], not 【2†L1-L5】, not [Source 2], not (2). Do not append line numbers or file names to a marker.
4. NO INVENTION. Never introduce a drug name, dose, threshold, percentage, or guideline recommendation that does not appear verbatim in the excerpts.
5. DISAGREEMENT. If excerpts conflict, report the conflict and cite both sides rather than choosing one.
6. STYLE. Be concise and clinical. Prefer short paragraphs or bullets. Lead with the direct answer. Do not add disclaimers about being an AI, and do not repeat the question.
7. SCOPE. This is decision support for clinicians, not patient advice. Do not add treatment recommendations beyond what the excerpts state."""


def format_context(contexts: list[ScoredChunk], max_tokens_per_chunk: int = 700) -> str:
    """Render retrieved chunks as numbered, provenance-labelled excerpts."""
    blocks: list[str] = []
    for i, scored in enumerate(contexts, start=1):
        chunk = scored.chunk
        text = truncate_tokens(chunk.text, max_tokens_per_chunk)
        header = f"[{i}] {chunk.title} | page {chunk.page} | section: {chunk.section}"
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)


def build_answer_prompt(question: str, contexts: list[ScoredChunk]) -> str:
    if not contexts:
        return (
            f"QUESTION: {question}\n\n"
            "CONTEXT EXCERPTS:\n(none retrieved)\n\n"
            "No excerpts were retrieved. Reply exactly: "
            '"The provided documents do not contain information to answer this question."'
        )
    return (
        f"CONTEXT EXCERPTS:\n\n{format_context(contexts)}\n\n"
        f"---\n\nQUESTION: {question}\n\n"
        "Answer using only the excerpts above, citing each claim with its bracketed "
        "marker. If the excerpts do not answer the question, say so explicitly."
    )


# ---------------------------------------------------------------------------
# Module 9 - LLM-as-judge prompts
# ---------------------------------------------------------------------------

FAITHFULNESS_SYSTEM = """You are a strict evaluator of factual grounding in medical text.

You receive CONTEXT excerpts and an ANSWER. Decide what proportion of the answer's factual claims are directly supported by the context.

Method:
1. Break the answer into atomic factual claims. Ignore filler, restatements of the question, and citation markers.
2. Mark each claim SUPPORTED only if the context states it or directly entails it. Plausible-but-absent means NOT supported.
3. An explicit refusal ("the documents do not contain...") counts as fully faithful.

Respond with ONLY a JSON object:
{"score": <0.0-1.0>, "supported": <int>, "total": <int>, "unsupported_claims": ["..."], "reason": "<one sentence>"}"""

ANSWER_RELEVANCE_SYSTEM = """You judge whether an ANSWER actually addresses the QUESTION asked.

Score only relevance and directness, NOT factual accuracy:
1.0 = fully answers exactly what was asked
0.5 = partially answers, or answers a related but different question
0.0 = does not address the question

A refusal is scored 1.0 if the question is genuinely unanswerable from the documents, otherwise 0.0.

Respond with ONLY a JSON object:
{"score": <0.0-1.0>, "reason": "<one sentence>"}"""

CONTEXT_RELEVANCE_SYSTEM = """You judge retrieval quality.

Given a QUESTION and numbered CONTEXT excerpts, decide which excerpts contain information useful for answering it.

Respond with ONLY a JSON object:
{"score": <fraction of excerpts that are relevant, 0.0-1.0>, "relevant_markers": [<int>, ...], "reason": "<one sentence>"}"""


def build_faithfulness_prompt(answer: str, contexts: list[ScoredChunk]) -> str:
    return f"CONTEXT:\n\n{format_context(contexts)}\n\n---\n\nANSWER:\n{answer}"


def build_answer_relevance_prompt(question: str, answer: str) -> str:
    return f"QUESTION: {question}\n\nANSWER:\n{answer}"


def build_context_relevance_prompt(question: str, contexts: list[ScoredChunk]) -> str:
    return f"QUESTION: {question}\n\nCONTEXT:\n\n{format_context(contexts, 400)}"


# ---------------------------------------------------------------------------
# Evaluation-set generation (bootstrapping a labelled test set from the corpus)
# ---------------------------------------------------------------------------

QUESTION_GEN_SYSTEM = """You write evaluation questions for a clinical retrieval system.

Given an excerpt from a medical document, write ONE question that a clinician might realistically ask and that this excerpt answers specifically.

CRITICAL — the question must test retrieval, not string matching:
- PARAPHRASE. Do not reuse the excerpt's distinctive phrases. Express the idea in different words, the way a clinician would ask it from memory rather than while reading the text.
- Prefer everyday clinical phrasing over the excerpt's technical wording (ask about "blood thinners" where the text says "direct oral anticoagulants", "kidney function" for "estimated glomerular filtration rate").
- Keep the condition or drug name if it is essential to make the question answerable, but do not copy the surrounding descriptive language.
- Be SELF-CONTAINED. Never write "this study", "the above", "the document", "these patients". Someone who has not seen the excerpt must be able to understand the question.
- Ask about one thing. Do not stack several sub-questions together.

Respond with ONLY a JSON object:
{"question": "...", "expected_information": "<one sentence on what a correct answer must contain>", "category": "<treatment|diagnosis|risk_factors|mechanism|dosage|adverse_effects|definition>"}"""


def build_question_gen_prompt(chunk_text: str, title: str, section: str) -> str:
    return (
        f"DOCUMENT: {title}\nSECTION: {section}\n\n"
        f"EXCERPT:\n{truncate_tokens(chunk_text, 600)}"
    )
