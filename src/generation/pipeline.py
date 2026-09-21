"""Module 7 + 8 - the end-to-end RAG pipeline.

    question -> query processing -> retrieval (top 20) -> rerank (top 5)
             -> prompt -> LLM -> citation extraction -> Answer

The pipeline is configured by ``RAGConfig`` rather than hard-wired, so the
exact same object serves the API, the UI, and every arm of the Module 9
comparison (vector vs hybrid vs hybrid+reranker).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import Answer, Citation, ScoredChunk, Timings, TokenUsage
from src.common.text import snippet
from src.embeddings.embedder import Embedder, get_embedder
from src.generation.llm import LLMClient, get_llm
from src.generation.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from src.retrieval.hybrid_search import make_retriever
from src.retrieval.index import RagIndex
from src.retrieval.query_processing import looks_out_of_scope, normalize_question
from src.retrieval.reranker import make_reranker

log = get_logger(__name__)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
REFUSAL_MARKER = "do not contain information"

# Models do not all emit the bracket format the prompt asks for. gpt-oss in
# particular is trained on OpenAI's file-citation tokens and produces
# "【1†L31-L33】", which silently parsed as zero citations and flipped
# well-grounded answers to grounded=False. Rather than trusting one format,
# normalise the common variants to "[n]" before extraction.
_CITE_CJK = re.compile(r"【\s*(\d{1,2})\s*(?:†[^】]*)?】")
_CITE_GROUPED = re.compile(r"\[\s*(\d{1,2}(?:\s*,\s*\d{1,2})+)\s*\]")
_CITE_SOURCE_WORD = re.compile(r"\[\s*(?:source|ref|citation)\s*[: ]\s*(\d{1,2})\s*\]", re.I)


def normalize_citation_markers(text: str) -> str:
    """Rewrite alternative citation syntaxes into the canonical ``[n]`` form."""
    text = _CITE_CJK.sub(lambda m: f"[{m.group(1)}]", text)
    text = _CITE_SOURCE_WORD.sub(lambda m: f"[{m.group(1)}]", text)
    # "[1, 3]" -> "[1][3]"
    text = _CITE_GROUPED.sub(
        lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), text
    )
    return text


@dataclass
class RAGConfig:
    strategy: str = "hybrid"          # vector | bm25 | hybrid | hybrid_weighted
    rerank: bool = True
    top_k: int = 20                   # candidates from retrieval
    top_n: int = 5                    # passages actually sent to the LLM
    temperature: float = 0.0
    max_tokens: int = 800
    alpha: float | None = None        # weighted-fusion mix, if used

    @property
    def label(self) -> str:
        base = self.strategy
        return f"{base}+rerank" if self.rerank else base


class RAGPipeline:
    def __init__(
        self,
        index: RagIndex,
        config: RAGConfig | None = None,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        settings = get_settings()
        self.index = index
        self.config = config or RAGConfig(
            top_k=settings.retrieval_top_k, top_n=settings.rerank_top_n
        )
        self.embedder = embedder or get_embedder(index.manifest.get("embedding_model"))
        # Resolved on first use. Retrieval-only callers (the chunk sweep, the
        # /retrieve endpoint, retrieval-only evaluation) never generate text
        # and should not need a provider configured at all.
        self._llm = llm

        kwargs = {}
        if self.config.alpha is not None:
            kwargs["alpha"] = self.config.alpha
        self.retriever = make_retriever(
            self.config.strategy, index, embedder=self.embedder, top_k=self.config.top_k, **kwargs
        )
        self.reranker = make_reranker("cross-encoder" if self.config.rerank else "none")

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def warmup(self) -> None:
        """Force the lazy models to load before any timed query.

        The embedder and cross-encoder load on first use, which would
        otherwise be billed to whichever query happened to arrive first —
        several seconds of model loading reported as "retrieval latency".
        Call this at service startup and before benchmarking.
        """
        t0 = time.perf_counter()
        self.embedder.encode_query("warmup")
        if isinstance(self.reranker, type(self.reranker)) and hasattr(self.reranker, "model"):
            try:
                self.reranker.model.predict([("warmup", "warmup passage")])
            except Exception as exc:  # a failed warmup must never block startup
                log.debug("reranker warmup skipped: %s", exc)
        log.info("models warm in %.1fs", time.perf_counter() - t0)

    # -- retrieval only (used by the eval harness and /retrieve) -------------

    def retrieve(self, question: str) -> tuple[list[ScoredChunk], float, float]:
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(question, top_k=self.config.top_k)
        retrieval_latency = time.perf_counter() - t0

        t1 = time.perf_counter()
        contexts = self.reranker.rerank(question, candidates, top_n=self.config.top_n)
        rerank_latency = time.perf_counter() - t1

        return contexts, retrieval_latency, rerank_latency

    # -- full pipeline -------------------------------------------------------

    def answer(self, question: str) -> Answer:
        started = time.perf_counter()
        question = normalize_question(question)

        if looks_out_of_scope(question):
            return Answer(
                question=question,
                answer="Please enter a clinical question.",
                confidence=0.0,
                grounded=False,
                strategy=self.config.label,
                model=self.llm.model,
                timings=Timings(total_latency=time.perf_counter() - started),
            )

        contexts, retrieval_latency, rerank_latency = self.retrieve(question)

        prompt = build_answer_prompt(question, contexts)
        response = self.llm.complete(
            ANSWER_SYSTEM_PROMPT,
            prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        answer_text = normalize_citation_markers(response.text)
        citations, cited_markers = extract_citations(answer_text, contexts)
        refused = REFUSAL_MARKER in answer_text.lower()
        confidence = compute_confidence(answer_text, contexts, cited_markers, refused)

        total = time.perf_counter() - started
        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            contexts=contexts,
            confidence=confidence,
            grounded=not refused and bool(citations),
            strategy=self.config.label,
            model=f"{response.provider}:{response.model}",
            timings=Timings(
                retrieval_latency=round(retrieval_latency, 4),
                rerank_latency=round(rerank_latency, 4),
                generation_latency=round(response.latency, 4),
                total_latency=round(total, 4),
            ),
            usage=TokenUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                estimated_cost_usd=round(response.cost_usd, 8),
            ),
        )


# ---------------------------------------------------------------------------
# Module 8 - citation extraction
# ---------------------------------------------------------------------------


def extract_citations(
    answer_text: str, contexts: list[ScoredChunk]
) -> tuple[list[Citation], set[int]]:
    """Map the ``[n]`` markers the model emitted back onto real passages.

    Markers outside the supplied range are dropped rather than rendered: a
    citation pointing at a document that was never retrieved is worse than no
    citation, because it looks authoritative in the UI.
    """
    markers = {int(m) for m in _CITATION_RE.findall(answer_text)}
    valid = {m for m in markers if 1 <= m <= len(contexts)}

    if invalid := markers - valid:
        log.warning("model cited non-existent markers: %s", sorted(invalid))

    citations: list[Citation] = []
    for marker in sorted(valid):
        scored = contexts[marker - 1]
        chunk = scored.chunk
        citations.append(
            Citation(
                marker=marker,
                document_id=chunk.document_id,
                document=chunk.source,
                title=chunk.title,
                page=chunk.page,
                section=chunk.section,
                snippet=snippet(chunk.text, 400),
                url=chunk.url,
                score=round(scored.rerank_score or scored.score, 4),
            )
        )
    return citations, valid


def compute_confidence(
    answer_text: str,
    contexts: list[ScoredChunk],
    cited_markers: set[int],
    refused: bool,
) -> float:
    """A transparent 0-1 confidence blending three observable signals.

    This is a heuristic, not a calibrated probability, and it is deliberately
    simple enough to explain: how strong was retrieval, how much of the answer
    is cited, and did the model lean on more than a single passage.
    """
    if refused or not contexts:
        return 0.0

    # 1. Retrieval strength. Cross-encoder logits are unbounded, so squash them.
    top = contexts[0]
    if top.rerank_score is not None:
        retrieval_signal = 1.0 / (1.0 + math.exp(-top.rerank_score))
    else:
        retrieval_signal = max(0.0, min(1.0, top.score))

    # 2. Citation density: fraction of substantive sentences carrying a marker.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer_text) if len(s.split()) >= 4]
    if sentences:
        cited = sum(1 for s in sentences if _CITATION_RE.search(s))
        citation_signal = cited / len(sentences)
    else:
        citation_signal = 0.0

    # 3. Corroboration: several distinct sources beats one.
    support_signal = min(len(cited_markers) / 3.0, 1.0)

    score = 0.5 * retrieval_signal + 0.35 * citation_signal + 0.15 * support_signal
    return round(max(0.0, min(1.0, score)), 3)


def build_pipeline(
    index_name: str = "default", config: RAGConfig | None = None
) -> RAGPipeline:
    return RAGPipeline(RagIndex.load(index_name), config=config)
