"""Tokenisation and small text helpers shared across the pipeline."""

from __future__ import annotations

import functools
import re
import unicodedata

import tiktoken

_ENCODING = "cl100k_base"


@functools.lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


def encode(text: str) -> list[int]:
    return _encoder().encode(text, disallowed_special=())


def decode(tokens: list[int]) -> str:
    return _encoder().decode(tokens)


def truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = encode(text)
    if len(tokens) <= max_tokens:
        return text
    return decode(tokens[:max_tokens])


_WS_RE = re.compile(r"[ \t\u00a0]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


_SENTENCE_RE = re.compile(
    r"""
    (?<![A-Z][a-z]\.)          # not an initial like "Dr."
    (?<!\b[A-Z]\.)             # not a single-letter abbreviation
    (?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\bet\sal\.)
    (?<!\bFig\.)(?<!\bNo\.)(?<!\bapprox\.)
    (?<=[.!?])\s+(?=[A-Z0-9(])
    """,
    re.VERBOSE,
)


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter tuned to avoid breaking on medical abbreviations."""
    parts = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# Kept deliberately small: clinical text is full of short but meaningful tokens.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from has have had he her his i if in into is it its
    of on or our s she t that the their them then there these they this to was were what when
    which who will with would you your
    """.split()
)


def tokenize_words(text: str, remove_stopwords: bool = True) -> list[str]:
    """Word tokenizer used by BM25. Lowercases and strips punctuation.

    Hyphenated compounds are split into their parts, the same way Lucene's
    standard tokenizer does. Medical writing is inconsistent about them
    ("first-line" vs "first line", "beta-blocker" vs "beta blocker"), and
    splitting on both the indexing and query side makes the two forms produce
    identical token sequences.

    Emitting the joined form *as well* was tried and removed: it makes a
    hyphenated query term contribute two or three times its share of IDF,
    which lets a passage rich in "first-line" outrank one that is actually
    about the condition being asked for.
    """
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if "-" in token or "'" in token:
            tokens.extend(p for p in re.split(r"[-']", token) if p)
        else:
            tokens.append(token)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def snippet(text: str, max_chars: int = 320) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
