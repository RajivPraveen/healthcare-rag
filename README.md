# Healthcare RAG Intelligence Platform

<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="FAISS" src="https://img.shields.io/badge/FAISS-012169?logo=meta&logoColor=white">
  <img alt="Groq" src="https://img.shields.io/badge/LLM-Groq%20gpt--oss--120b-F55036">
</p>

Ask a clinical question. Get an answer written **only from retrieved papers and drug labels**, with the document, page, and section next to every claim. If the library does not contain the answer, the system says so.

<p align="center">
  <img src="docs/images/ask.png" alt="Grounded answer with citations for a hypertension question" width="100%">
</p>

| | |
|---|---|
| **Corpus** | 289 PubMed Central papers and FDA drug labels, 8,754 chunks |
| **Retrieval** | Vector, BM25, hybrid (reciprocal rank fusion), cross-encoder rerank |
| **Best setting** | Hybrid + reranker · MRR **0.965** · Recall@1 **94.7%** on the low-leakage slice |
| **Answer quality** | Faithfulness **0.91** · citation accuracy **1.0** · about **1.6s** per question |

<details>
<summary><b>Corpus browser</b> — what is actually indexed</summary>
<p align="center">
  <img src="docs/images/corpus.png" alt="Corpus tab showing 289 documents and 8754 chunks" width="100%">
</p>
</details>

<details>
<summary><b>Evaluation dashboard</b> — vector vs BM25 vs hybrid vs hybrid + reranker</summary>
<p align="center">
  <img src="docs/images/evaluation.png" alt="Evaluation chart where hybrid plus reranker leads on MRR" width="100%">
</p>
</details>

```
Q: What are the recommended first-line treatments for hypertension?

A: - The WHO Essential Medicines List identifies ACE inhibitors, calcium-channel
     blockers, angiotensin-receptor blockers, and thiazide diuretics as medicines
     for the pharmacological management of hypertension [3].
   - Guideline-recommended initial regimens often use dual therapy, for example an
     ACE inhibitor combined with a thiazide diuretic [1].

Sources
  [1] Retrospective comparison of ChatGPT-4 treatment recommendations…  p.1  Abstract
  [3] Hypertension Pharmacological Treatment in Adults: A WHO Guideline  p.8  Body

hybrid+rerank · confidence 0.94 · retrieval 0.04s · rerank 0.14s · generation 0.92s
```

The point of the project isn't that it answers questions — it's that every design choice is **measured** rather than assumed: three retrieval strategies compared head to head, four chunk sizes swept, and hallucination tracked explicitly through refusal rate and citation accuracy.

---

## Architecture

```mermaid
flowchart TD
    docs["PubMed Central + openFDA + your PDFs"] --> ingest["Ingest, clean, chunk, keep page and section"]
    ingest --> embed["Local embeddings · BGE-small"]
    embed --> faiss["Vector search · FAISS"]
    ingest --> bm25["Keyword search · BM25"]
    faiss --> hybrid["Hybrid · reciprocal rank fusion"]
    bm25 --> hybrid
    hybrid --> rerank["Cross-encoder reranker · top 20 to top 5"]
    rerank --> llm["Groq gpt-oss-120b"]
    llm --> answer["Answer + citations + confidence"]
    answer --> ui["FastAPI → Streamlit"]
```

---

## Quickstart

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env        # add a GROQ_API_KEY (free) — optional, see below

hcrag ingest                # download ~290 documents
hcrag index                 # chunk + embed + build FAISS & BM25
hcrag ask "What are the first-line treatments for hypertension?"
```

Then the app:

```bash
hcrag serve      # FastAPI  → http://localhost:8000/docs
hcrag ui         # Streamlit → http://localhost:8501
```

Or the whole stack:

```bash
docker compose --profile setup run --rm ingest   # one-off corpus + index
docker compose up --build                        # qdrant + api + ui
```

### LLM providers

`LLM_PROVIDER=auto` resolves in order: **Groq → Gemini → OpenAI → Ollama → extractive**.

| Provider | Cost | Notes |
|---|---|---|
| Groq | free tier | `openai/gpt-oss-120b`; 1k req/day, 8k tokens/min |
| Gemini | free tier | `gemini-2.5-flash` |
| Ollama | free, local | unmetered — best for bulk evaluation runs |
| *extractive* | free, offline | **no key required**; the repo is fully runnable without credentials |

Embeddings and reranking always run locally, so the LLM is the only component that ever touches a paid service.

---

## The modules

| | Module | Where |
|---|---|---|
| 1 | Data collection | `src/ingestion/downloader.py` |
| 2 | Extraction, cleaning, chunking | `src/ingestion/{loader,cleaner,chunker}.py` |
| 3 | Chunk-size experiment | `hcrag sweep` |
| 4 | Embeddings | `src/embeddings/embedder.py` |
| 5 | Vector / BM25 / Hybrid retrieval | `src/retrieval/` |
| 6 | Cross-encoder reranking | `src/retrieval/reranker.py` |
| 7 | RAG generation | `src/generation/pipeline.py` |
| 8 | Citations | `extract_citations()` |
| 9 | Evaluation | `src/evaluation/` |
| 10 | Experiment dashboard | `hcrag compare` + Streamlit tab |
| 11 | FastAPI backend | `api/main.py` |
| 12 | Streamlit frontend | `frontend/app.py` |
| 13 | Deployment | `Dockerfile`, `docker-compose.yml` |

---

## Corpus

Built from two public, licence-clean sources, downloaded programmatically and idempotently:

- **PubMed Central Open Access** — clinical review articles across 30 conditions, fetched as JATS XML so the real section hierarchy survives into chunk metadata.
- **openFDA drug labels** — structured product labelling, where each field (*Indications*, *Contraindications*, *Adverse Reactions*…) becomes a section.

Searching PMC required care. A relevance-sorted free-text query for `hypertension management` returns epidemiology papers that merely *mention* hypertension; requiring the concept in the **title** is what makes the corpus clinically on-topic.

---

## Engineering notes

Six decisions that were not obvious, and what forced them.

**BM25 is hand-written, not `rank_bm25`.** The scoring internals are the reason the lexical arm exists, and two fixes came directly out of reading them. Hyphenated terms are split (`first-line` → `first`, `line`) on *both* sides, matching Lucene, because indexing the joined form too made a hyphenated query term contribute triple its share of IDF. And a **coordination factor** scales each score by the fraction of distinct query terms matched — without it, plain BM25 ranked a depression-guideline passage dense in "first line … recommended" above every hypertension document, because it matched three common terms many times while missing the one discriminative term. There's a regression test for exactly that.

**Hybrid fuses ranks, not scores.** Cosine similarity is bounded around 0.6–0.9; BM25 is unbounded and corpus-dependent. Reciprocal Rank Fusion sidesteps the incomparable scales entirely. Weighted score fusion is implemented too, but it needs per-query normalisation and is sensitive to the candidate pool.

**The reranker enforces source diversity.** Chunk overlap plus one long relevant section let a single document occupy every top slot — three near-duplicate passages, one cited source. A per-document quota backfills from the next-best document instead.

**Rate limiting is token-aware.** Groq's free tier allows 1000 requests/day but only **8000 tokens/minute**, and one RAG call with five passages costs ~3300 tokens. A request-per-minute throttle would sail straight past the limit. The limiter tracks a rolling 60-second token window and blocks until there's room. Buckets are per-model, so generation (120b) and the eval judge (20b) draw on separate budgets.

**Reasoning models need an effort cap.** `gpt-oss` spends its completion budget on hidden chain-of-thought and returns an **empty message** when `max_tokens` runs out — which looks exactly like a JSON parse failure. Token accounting gave it away (300 completion tokens, zero content). `reasoning_effort=low` keeps the visible answer inside the budget.

**Models are warmed at startup.** The embedder and cross-encoder load lazily, so without a warmup the first query reports ~5 s of "retrieval latency" that is really model loading. Every latency number below is post-warmup.

---

## Evaluation

```bash
hcrag make-evalset --n 90   # bootstrap a labelled test set
hcrag evaluate              # retrieval + generation + engineering metrics
hcrag compare               # Vector vs BM25 vs Hybrid vs Hybrid+Reranker
hcrag sweep                 # 300 / 500 / 800 / 1200-token chunking
```

**Retrieval** — Recall@K, Precision@K, MRR, nDCG@K, Hit Rate, MAP, at both chunk and document granularity.
**Generation** — faithfulness, answer relevance, context relevance (LLM-judged, with a *separate* model from the generator), plus citation accuracy and coverage computed deterministically.
**Engineering** — per-stage latency, p95, tokens, cost per query.

### Measured results

289 documents / 8,754 chunks, `openai/gpt-oss-120b` on Groq, judged by `gpt-oss-20b`.

The test set is 76 questions: 63 generated from a known passage (these carry gold chunk labels and are what retrieval is scored on), 10 hand-written clinical questions, and 3 deliberately out-of-scope questions used to check that the system refuses rather than invents.

| Strategy | Recall@1 | Recall@5 | MRR | nDCG@5 | Retrieval time |
|---|---|---|---|---|---|
| Vector only | 0.841 | 0.968 | 0.900 | 0.917 | 7 ms |
| BM25 only | 0.905 | 1.000 | 0.943 | 0.958 | <1 ms |
| Hybrid (RRF) | 0.873 | 0.984 | 0.925 | 0.940 | 8 ms |
| **Hybrid + Reranker** | **0.936** | 0.984 | **0.955** | **0.962** | 138 ms |

**Do not read this table on its own.** BM25 scoring a perfect Recall@5 against a dense retriever is not a plausible result, and chasing it down is the most interesting thing in this project — see [Where the evaluation misled](#where-the-evaluation-misled-and-what-fixed-it) below. On the questions that are not contaminated, BM25's lead disappears entirely.

Generation, measured on Hybrid + Reranker:

| Metric | Value |
|---|---|
| Faithfulness (judged) | 0.911 |
| Answer relevance (judged) | 0.889 |
| Context relevance (judged) | 0.562 |
| Citation accuracy | **1.000** |
| Citation coverage | 0.920 |
| False refusal rate | 0.080 |
| Mean end-to-end latency | 1.64 s |
| Tokens per query | ~2,700 |
| Cost per query | $0 (free tier) |

Citation accuracy of 1.000 means every marker the model emitted pointed at a passage that was actually retrieved — no invented references. That is checked programmatically, not judged.

Context relevance at 0.562 is the weakest number and the honest read is that it should be: five passages are sent to the model and only two or three typically bear on the question. Reducing N would raise it at the cost of recall. Faithfulness stays high because the model ignores the irrelevant passages rather than being misled by them.

### Chunk size (Module 3)

| Target tokens | Chunks | Mean actual | Recall@5 | MRR | nDCG@5 | Query |
|---|---|---|---|---|---|---|
| **300** | 13,134 | 240 | 0.968 | **0.940** | **0.948** | 10.4 ms |
| 500 | 8,754 | 335 | **0.984** | 0.925 | 0.940 | 9.2 ms |
| 800 | 6,841 | 408 | 0.968 | 0.927 | 0.937 | 11.4 ms |
| 1200 | 6,011 | 449 | 0.968 | 0.919 | 0.932 | 9.8 ms |

Smaller chunks rank the right passage higher (MRR 0.940 at 300 vs 0.919 at 1200) because a short chunk is *about* one thing, so its embedding is not diluted. 500 tokens edges ahead on Recall@5 while halving the index. The spread is narrow, which is itself the finding: chunk size is worth about two points of MRR here, far less than the choice of retrieval strategy.

Actual chunk sizes land well below target (240 for a 300-token target) because chunks stop at sentence and section boundaries rather than mid-claim — a deliberate trade of packing efficiency for citations that point at a complete thought.

### How the test set is built

Questions come from two places. **Synthetic + labelled**: sample chunks across the corpus, have an LLM write a question answerable from that chunk, and treat that chunk and its document as ground truth. **Curated + unlabelled**: hand-written questions, including deliberately unanswerable ones ("the ACME-9000 trial of zolpidextrin") that test whether the system *refuses* instead of inventing.

Generated questions are filtered for self-containment — "What did this study find?" measures nothing, because no retriever can resolve "this study".

### The most interesting result: the benchmark was measuring the wrong thing

The first comparison run said **BM25 was the best retriever**, with a perfect Recall@5 of 1.000 — beating both dense and hybrid retrieval. That is the opposite of what the architecture predicts, and perfect scores are a warning sign, not a victory.

The cause was **lexical leakage**. A question generated *from* a passage reuses that passage's vocabulary, so BM25 matches it by string overlap rather than by understanding the question. Measuring the fraction of question terms appearing verbatim in the gold chunk showed a mean overlap of **0.64**, with 83% of questions above 0.45 and one question at **1.00** — every single term copied from its source.

Filtering aggressively was the wrong fix, because some overlap is unavoidable and legitimate: a question about amlodipine has to say "amlodipine". So leakage is *scored and stored* on every question, and results are **stratified**:

| Slice | Strategy | Recall@1 | MRR |
|---|---|---|---|
| **High leakage** (n=44) | Vector | 0.841 | 0.898 |
| | **BM25** | **0.932** | **0.960** |
| | Hybrid + Reranker | 0.932 | 0.951 |
| **Low leakage** (n=19) | Vector | 0.842 | 0.905 |
| | BM25 | 0.842 | 0.904 |
| | **Hybrid + Reranker** | **0.947** | **0.965** |

On leaky questions BM25 beats dense retrieval by 9 points of Recall@1. On clean questions its advantage **disappears entirely** — BM25 and vector are identical to three decimal places — and hybrid + reranking is the clear winner, which is what the architecture predicted all along.

Two things follow. BM25's apparent superiority was an artifact of how the test set was built, and the arm selected for expensive generation metrics is therefore ranked on the low-leakage slice, not the aggregate.

### Remaining limitations, stated plainly

**Recall@5 is at the ceiling (≈1.0) for every strategy.** With 289 documents on topically distinct conditions, identifying the right *document* from a question that names the drug or condition is simply easy. The benchmark discriminates at Recall@1 and MRR, not at Recall@5. A larger, more topically overlapping corpus would be needed to stress it properly.

**Synthetic labels mark only the source chunk as relevant**, so a retriever that surfaces an equally correct passage from a different paper is scored as wrong. Absolute recall is under-reported; the bias is identical across arms, so the comparison holds.

**Generation metrics run on a 25-question subsample.** Groq's free tier allows 1000 requests/day, and four arms × 76 questions × (1 generation + 3 judge calls) exceeds that. Retrieval is scored on all 76 questions for all arms because it needs no LLM at all.

---

## API

```
POST /query          question → grounded answer + citations + timings
POST /retrieve       retrieval only, no LLM — inspect ranking cheaply
POST /documents      upload a PDF (reindexes in the background)
GET  /documents      list the indexed corpus
GET  /health         component + provider status
GET  /metrics        rolling latency percentiles, tokens, cost
GET  /evaluation     latest experiment results
```

```bash
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"What are the contraindications for ACE inhibitors?"}'
```

---

## Layout

```
data/{raw,processed,indexes,metadata,evaluation}
notebooks/    01_data_exploration · 02_chunking · 03_retrieval · 04_rag_evaluation
src/
  ingestion/  downloader · loader · cleaner · chunker
  embeddings/ embedder
  retrieval/  vector_search · bm25 · hybrid_search · reranker · index · query_processing
  generation/ prompts · llm · pipeline
  evaluation/ retrieval_metrics · rag_evaluation · dataset · experiments
api/main.py · frontend/app.py · tests/
```

```bash
pytest -q        # 64 tests
ruff check .
```

---

## Safety

Decision support for exploring literature, **not** a medical device and not patient advice. The prompt forbids using prior knowledge, requires a citation per claim, and mandates refusal when the context is insufficient — but an LLM can still misread a passage. Every answer ships with its sources so claims can be checked against the original text.
