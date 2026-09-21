"""Module 9 (part 2) - building the evaluation set.

Hand-labelling hundreds of clinical questions is not realistic for a solo
project, so the test set is built two ways and the two are kept distinct:

1. **Synthetic, labelled.** Sample chunks across the corpus, have an LLM write
   a question answerable *from that chunk*, and treat that chunk (and its
   document) as ground truth. This is what makes Recall/MRR/nDCG computable.

2. **Curated, unlabelled.** Hand-written questions including deliberately
   unanswerable ones. These have no gold chunk, so they are excluded from
   retrieval metrics, but they drive the generation metrics and — importantly
   — test whether the system *refuses* instead of inventing an answer.

Known limitation, stated plainly because it affects how the numbers should be
read: synthetic labels mark only the source chunk as relevant, so a retriever
that surfaces an equally correct passage from elsewhere is scored as wrong.
Recall is therefore under-reported. Document-level granularity softens this,
and it biases all strategies equally, so the *comparison* between them stays
sound even though the absolute values are pessimistic.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import Chunk, EvalQuestion
from src.common.text import tokenize_words
from src.generation.llm import get_judge_llm
from src.generation.prompts import QUESTION_GEN_SYSTEM, build_question_gen_prompt
from src.retrieval.index import RagIndex

log = get_logger(__name__)

# Hand-written probes. The unanswerable ones are the important half: a RAG
# system that never refuses is not grounded, it is just fluent.
CURATED_QUESTIONS: list[dict] = [
    {
        "question": "What are the recommended first-line treatments for hypertension?",
        "expected_information": "First-line antihypertensive drug classes and when they are used.",
        "category": "treatment",
    },
    {
        "question": "Which medications are used as first-line therapy for type 2 diabetes?",
        "expected_information": "Initial pharmacological therapy for type 2 diabetes.",
        "category": "treatment",
    },
    {
        "question": "What are the main contraindications for ACE inhibitors?",
        "expected_information": "Situations in which ACE inhibitors must not be used.",
        "category": "contraindications",
    },
    {
        "question": "What are the most common adverse reactions associated with amlodipine?",
        "expected_information": "Adverse reactions reported for amlodipine.",
        "category": "adverse_effects",
    },
    {
        "question": "How is sepsis defined and what are the early management priorities?",
        "expected_information": "Definition of sepsis and initial resuscitation steps.",
        "category": "definition",
    },
    {
        "question": "What are the major risk factors for chronic kidney disease?",
        "expected_information": "Conditions and exposures that increase CKD risk.",
        "category": "risk_factors",
    },
    {
        "question": "What is the mechanism of action of SGLT2 inhibitors?",
        "expected_information": "How SGLT2 inhibitors lower glucose.",
        "category": "mechanism",
    },
    {
        "question": "Which anticoagulants are recommended for stroke prevention in atrial fibrillation?",
        "expected_information": "Anticoagulant options for AF stroke prevention.",
        "category": "treatment",
    },
    {
        "question": "What inhaled therapies are used for long-term asthma control?",
        "expected_information": "Controller inhaler classes for asthma.",
        "category": "treatment",
    },
    {
        "question": "What is the recommended dosing of lisinopril for hypertension?",
        "expected_information": "Starting and maintenance dosing for lisinopril.",
        "category": "dosage",
    },
    # --- deliberately unanswerable -----------------------------------------
    {
        "question": "What is the current stock price of Pfizer?",
        "expected_information": "Not present in a clinical corpus; the system must refuse.",
        "category": "out_of_scope",
        "unanswerable": True,
    },
    {
        "question": "What did the 2027 WHO guideline recommend for treating Martian fever?",
        "expected_information": "A fabricated condition and future guideline; the system must refuse.",
        "category": "out_of_scope",
        "unanswerable": True,
    },
    {
        "question": "How many patients were enrolled in the ACME-9000 trial of zolpidextrin?",
        "expected_information": "A fabricated drug and trial; the system must refuse.",
        "category": "out_of_scope",
        "unanswerable": True,
    },
]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict | None:
    """LLMs wrap JSON in prose or fences more often than they should."""
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# Anaphoric references that make a question unanswerable without already
# knowing which passage it came from. A retriever cannot resolve "this study",
# so such a question measures nothing and must be rejected.
_ANAPHORIC = re.compile(
    r"\b(this|these|that|the)\s+"
    # Allow intervening adjectives: the real generator produced "this
    # systematic scoping review", which a determiner-adjacent pattern misses.
    r"(?:[a-z]+(?:ic|al|ive|ed|ing|ary|ent)\s+){0,3}"
    r"(study|studies|review|reviews|paper|article|trial|trials|document|report|"
    r"excerpt|passage|guideline|text|analysis|work|research|figure|table|author|"
    r"cohort|dataset|survey)s?\b",
    re.I,
)


def lexical_overlap(question: str, chunk_text: str) -> float:
    """Fraction of the question's content words that appear in its source chunk.

    This is the leakage metric for a synthetic evaluation set. A question
    generated *from* a passage tends to reuse that passage's distinctive
    vocabulary, and a lexical retriever then matches it by string overlap
    rather than by understanding the question. Left unchecked it produced a
    benchmark where BM25 scored a perfect Recall@5 — not because BM25 is
    excellent, but because the questions were near-copies of their answers.

    Only near-verbatim copies are filtered out (the default threshold is
    deliberately loose), because *some* overlap is unavoidable and legitimate:
    a question about amlodipine has to say "amlodipine". The score is stored
    on every question instead, so results can be reported separately for a
    low-leakage slice — see ``stratify_by_leakage``.
    """
    q_terms = set(tokenize_words(question))
    if not q_terms:
        return 0.0
    chunk_terms = set(tokenize_words(chunk_text))
    return len(q_terms & chunk_terms) / len(q_terms)


def is_self_contained(question: str) -> bool:
    """Reject questions that only make sense next to their source passage."""
    if len(question.split()) < 4:
        return False
    if _ANAPHORIC.search(question):
        return False
    return not re.search(r"\b(above|below|mentioned earlier|the following)\b", question, re.I)


def _sample_chunks(index: RagIndex, n: int, seed: int = 13) -> list[Chunk]:
    """Pick substantive chunks spread across distinct documents.

    One question per document keeps the evaluation set from over-representing
    whichever paper happens to be longest.
    """
    rng = random.Random(seed)
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in index.chunks:
        # Short or boilerplate-ish chunks make degenerate questions.
        if chunk.token_count < 180:
            continue
        if chunk.section.lower().startswith(("reference", "abbreviation")):
            continue
        by_doc.setdefault(chunk.document_id, []).append(chunk)

    doc_ids = list(by_doc)
    rng.shuffle(doc_ids)

    picked: list[Chunk] = []
    # Round-robin over documents so we exhaust breadth before depth.
    round_no = 0
    while len(picked) < n and doc_ids:
        progressed = False
        for doc_id in doc_ids:
            candidates = by_doc[doc_id]
            if round_no < len(candidates):
                picked.append(rng.choice(candidates))
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
        round_no += 1
    return picked[:n]


def generate_evaluation_set(
    n: int = 60,
    index_name: str = "default",
    out: Path | None = None,
    include_curated: bool = True,
    seed: int = 13,
    max_lexical_overlap: float = 0.75,
) -> Path:
    settings = get_settings()
    out = out or (settings.eval_dir / "questions.json")
    index = RagIndex.load(index_name)
    # Build the test set with the smaller/secondary model. Writing a question
    # from a passage is an easy task, and providers meter tokens per model, so
    # this leaves the primary model's budget intact for the evaluation itself.
    llm = get_judge_llm()

    if llm.provider == "extractive":
        log.warning(
            "No LLM available, so only the %d curated questions will be written. "
            "Set GROQ_API_KEY to generate the labelled synthetic set.",
            len(CURATED_QUESTIONS),
        )
        chunks: list[Chunk] = []
    else:
        chunks = _sample_chunks(index, n, seed=seed)
        log.info("generating %d questions with %s", len(chunks), llm.model)

    questions: list[EvalQuestion] = []
    rejected = 0
    leaky = 0

    for i, chunk in enumerate(chunks):
        prompt = build_question_gen_prompt(chunk.text, chunk.title, chunk.section)
        try:
            response = llm.complete(
                QUESTION_GEN_SYSTEM, prompt, temperature=0.3, max_tokens=300
            )
        except Exception as exc:
            log.warning("question generation failed for %s: %s", chunk.chunk_id, exc)
            continue

        payload = _parse_json(response.text)
        if not payload or not payload.get("question"):
            log.debug("unparseable generation output: %s", response.text[:120])
            rejected += 1
            continue

        question_text = payload["question"].strip()
        if not is_self_contained(question_text):
            log.debug("rejected non-self-contained question: %s", question_text[:90])
            rejected += 1
            continue

        overlap = lexical_overlap(question_text, chunk.text)
        if overlap > max_lexical_overlap:
            log.debug("rejected high-leakage question (%.2f): %s", overlap, question_text[:90])
            rejected += 1
            leaky += 1
            continue

        questions.append(
            EvalQuestion(
                question_id=f"syn_{i:04d}",
                question=question_text,
                expected_information=payload.get("expected_information", "").strip(),
                relevant_document_ids=[chunk.document_id],
                relevant_chunk_ids=[chunk.chunk_id],
                category=payload.get("category", "general"),
                source_title=chunk.title,
                lexical_overlap=round(overlap, 3),
            )
        )
        if (i + 1) % 10 == 0:
            log.info(
                "  %d/%d processed — %d kept, %d rejected",
                i + 1,
                len(chunks),
                len(questions),
                rejected,
            )

    if include_curated:
        for i, item in enumerate(CURATED_QUESTIONS):
            questions.append(
                EvalQuestion(
                    question_id=f"cur_{i:04d}",
                    question=item["question"],
                    expected_information=item["expected_information"],
                    category=item.get("category", "general"),
                    unanswerable=item.get("unanswerable", False),
                )
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([q.model_dump() for q in questions], indent=2), encoding="utf-8"
    )
    labelled = [q for q in questions if q.relevant_chunk_ids]
    mean_overlap = (
        sum(q.lexical_overlap for q in labelled) / len(labelled) if labelled else 0.0
    )
    log.info(
        "evaluation set: %d questions (%d labelled, %d curated) -> %s",
        len(questions),
        len(labelled),
        len(questions) - len(labelled),
        out,
    )
    log.info(
        "  rejected %d (%d for lexical leakage above %.2f); mean overlap of kept: %.2f",
        rejected,
        leaky,
        max_lexical_overlap,
        mean_overlap,
    )
    return out


def load_evaluation_set(path: Path | None = None) -> list[EvalQuestion]:
    path = path or (get_settings().eval_dir / "questions.json")
    if not path.exists():
        raise FileNotFoundError(
            f"No evaluation set at {path}. Create one with `hcrag make-evalset`."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion.model_validate(item) for item in payload]


def backfill_lexical_overlap(index, path: Path | None = None) -> Path:
    """Recompute and store leakage scores for an existing evaluation set."""
    path = path or (get_settings().eval_dir / "questions.json")
    questions = load_evaluation_set(path)
    for question in questions:
        if not question.relevant_chunk_ids:
            continue
        chunk = index.get(question.relevant_chunk_ids[0])
        if chunk is not None:
            question.lexical_overlap = round(lexical_overlap(question.question, chunk.text), 3)
    path.write_text(
        json.dumps([q.model_dump() for q in questions], indent=2), encoding="utf-8"
    )
    return path


def stratify_by_leakage(
    questions: list[EvalQuestion], threshold: float = 0.5
) -> dict[str, list[EvalQuestion]]:
    """Split labelled questions into low- and high-leakage slices.

    Aggregate metrics over the whole set are dominated by questions that
    share vocabulary with their source passage, which flatters lexical
    retrieval and pushes every strategy towards the ceiling. Reporting the
    low-leakage slice separately is what reveals the real difference between
    dense, lexical and hybrid retrieval.
    """
    labelled = [q for q in questions if q.relevant_document_ids]
    return {
        "all": labelled,
        "low_leakage": [q for q in labelled if q.lexical_overlap <= threshold],
        "high_leakage": [q for q in labelled if q.lexical_overlap > threshold],
    }
