"""Module 5, Version B - BM25 lexical retrieval.

Implemented directly rather than pulled from a library, because the scoring
behaviour is the whole point of having a lexical arm:

    score(q, d) = Σ_t  IDF(t) · ( tf(t,d) · (k1 + 1) ) / ( tf(t,d) + k1 · (1 - b + b · |d|/avgdl) )

Dense retrieval is weak exactly where clinical search is strongest: exact drug
names, dosages, ICD codes, and rare terms ("dapagliflozin", "HbA1c"). A
bi-encoder maps those to a fuzzy neighbourhood; BM25 matches them literally and
rewards their rarity through IDF. That is why the hybrid arm exists.

Scoring runs over an inverted index, so cost scales with query length rather
than corpus size.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.logging import get_logger
from src.common.schemas import Chunk
from src.common.text import tokenize_words

log = get_logger(__name__)


class BM25Index:
    """Okapi BM25 over an inverted index.

    Parameters
    ----------
    k1: term-frequency saturation. Higher = repeated terms keep adding score.
    b:  length normalisation. 1.0 fully normalises by document length, 0 ignores it.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, coord_lambda: float = 1.0) -> None:
        self.k1 = k1
        self.b = b
        self.coord_lambda = coord_lambda
        self.chunk_ids: list[str] = []
        self.doc_lengths: np.ndarray = np.zeros(0, dtype=np.float32)
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        # term -> (doc indices, term frequencies)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -- build ---------------------------------------------------------------

    def build(self, chunks: list[Chunk]) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        n_docs = len(chunks)
        if n_docs == 0:
            return

        raw_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(n_docs, dtype=np.float32)

        for doc_idx, chunk in enumerate(chunks):
            # Indexing title+section alongside body lets a query like
            # "amlodipine contraindications" match on metadata too.
            tokens = tokenize_words(f"{chunk.title} {chunk.section} {chunk.text}")
            lengths[doc_idx] = len(tokens)
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for term, tf in counts.items():
                raw_postings[term].append((doc_idx, tf))

        self.doc_lengths = lengths
        self.avgdl = float(lengths.mean()) if n_docs else 0.0

        self.postings = {}
        self.idf = {}
        for term, entries in raw_postings.items():
            doc_idx = np.array([e[0] for e in entries], dtype=np.int32)
            tfs = np.array([e[1] for e in entries], dtype=np.float32)
            self.postings[term] = (doc_idx, tfs)
            df = len(entries)
            # Robertson/Sparck-Jones IDF with the +1 smoothing that keeps it
            # non-negative for terms appearing in more than half the corpus.
            self.idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        log.info("BM25 index built: %d docs, %d unique terms", n_docs, len(self.postings))

    # -- search --------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.chunk_ids:
            return []
        # Distinct terms only. Query-side term frequency adds nothing for
        # short questions and double-counts words that repeat.
        terms = list(dict.fromkeys(tokenize_words(query)))
        if not terms:
            return []

        n_docs = len(self.chunk_ids)
        scores = np.zeros(n_docs, dtype=np.float32)
        matched = np.zeros(n_docs, dtype=np.float32)
        # Precompute the length-normalisation denominator component per doc.
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_lengths / max(self.avgdl, 1e-9))

        for term in terms:
            posting = self.postings.get(term)
            if posting is None:
                continue
            doc_idx, tfs = posting
            idf = self.idf[term]
            contribution = idf * (tfs * (self.k1 + 1.0)) / (tfs + norm[doc_idx])
            scores[doc_idx] += contribution
            matched[doc_idx] += 1.0

        # Coordination factor.
        #
        # Plain BM25 sums independently over terms, so a passage that matches
        # three moderately-weighted words many times can beat one that matches
        # every word including the rare, discriminative one. On "first-line
        # treatments for hypertension" that reliably surfaced passages
        # containing "first line ... recommended" and no hypertension at all.
        # Scaling by the fraction of distinct query terms present restores the
        # requirement that a good match covers the whole question.
        if self.coord_lambda > 0:
            coverage = matched / len(terms)
            scores *= coverage**self.coord_lambda

        nonzero = np.flatnonzero(scores)
        if nonzero.size == 0:
            return []
        k = min(top_k, nonzero.size)
        # argpartition then sort: O(n) selection instead of a full sort.
        top = nonzero[np.argpartition(-scores[nonzero], k - 1)[:k]]
        top = top[np.argsort(-scores[top])]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "bm25.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "coord_lambda": self.coord_lambda,
                    "chunk_ids": self.chunk_ids,
                    "doc_lengths": self.doc_lengths,
                    "avgdl": self.avgdl,
                    "idf": self.idf,
                    "postings": self.postings,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        (path / "bm25_meta.json").write_text(
            json.dumps({"k1": self.k1, "b": self.b, "n_docs": len(self.chunk_ids)}),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        with (path / "bm25.pkl").open("rb") as fh:
            state = pickle.load(fh)
        self.k1 = state["k1"]
        self.b = state["b"]
        self.coord_lambda = state.get("coord_lambda", 1.0)
        self.chunk_ids = state["chunk_ids"]
        self.doc_lengths = state["doc_lengths"]
        self.avgdl = state["avgdl"]
        self.idf = state["idf"]
        self.postings = state["postings"]

    @property
    def size(self) -> int:
        return len(self.chunk_ids)
