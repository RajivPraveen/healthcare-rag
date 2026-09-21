"""Module 2 (part 1) - text cleaning.

PDF and XML extraction leaves artefacts that wreck both embeddings and BM25:
de-hyphenated line wraps, running headers, bare page numbers, citation
brackets, and licence boilerplate. This module removes them while being
careful not to destroy clinically meaningful content (dosages, ranges, units).
"""

from __future__ import annotations

import re
from collections import Counter

from src.common.text import normalize_whitespace

# "hyper-\ntension" -> "hypertension"
_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")
# A line that is nothing but a page number, possibly decorated.
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s*)?[-–—|]?\s*\d{1,4}\s*[-–—|]?\s*$", re.I)
# Inline numeric citations: "[12]", "[3,4]", "[5-9]"
_NUMERIC_CITATION = re.compile(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]")
# Figure/table cross references left dangling after caption removal
_FIG_REF = re.compile(r"\((?:see\s+)?(?:fig(?:ure)?|table)\.?\s*\d+[a-z]?\)", re.I)
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Ligatures PyMuPDF sometimes emits
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "…": "..."}

_BOILERPLATE_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"^this is an open[- ]access article distributed.*$",
        r"^creative commons attribution.*$",
        r"^all rights reserved\.?$",
        r"^downloaded from .*$",
        r"^for personal use only.*$",
        r"^see discussions, stats, and author profiles.*$",
        r"^the copyright holder for this preprint.*$",
        r"^\s*doi:\s*\S+\s*$",
        r"^\s*issn[: ].*$",
    )
]


def _strip_ligatures(text: str) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str, drop_citations: bool = True) -> str:
    """Clean a single block of extracted text."""
    if not text:
        return ""

    text = _CONTROL.sub(" ", text)
    text = _strip_ligatures(text)
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)

    if drop_citations:
        text = _NUMERIC_CITATION.sub(" ", text)
        text = _FIG_REF.sub(" ", text)

    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _PAGE_NUMBER_LINE.match(stripped):
            continue
        if any(p.match(stripped) for p in _BOILERPLATE_PATTERNS):
            continue
        kept.append(stripped)

    return normalize_whitespace("\n".join(kept))


def detect_running_headers(pages: list[str], min_ratio: float = 0.5) -> set[str]:
    """Find header/footer lines that repeat across most pages of a document.

    Journal PDFs stamp the article title or journal name onto every page. Those
    lines are pure noise in a chunk and actively mislead BM25, so we find any
    short line appearing on at least ``min_ratio`` of pages and drop it.
    """
    if len(pages) < 4:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
        # Headers/footers live in the first and last few lines.
        candidates = lines[:3] + lines[-3:]
        for line in set(candidates):
            if 4 <= len(line) <= 120:
                counts[line] += 1

    threshold = max(2, int(len(pages) * min_ratio))
    return {line for line, n in counts.items() if n >= threshold}


def remove_lines(text: str, blacklist: set[str]) -> str:
    if not blacklist:
        return text
    kept = [ln for ln in text.split("\n") if ln.strip() not in blacklist]
    return normalize_whitespace("\n".join(kept))


def is_meaningful(text: str, min_chars: int = 120, min_alpha_ratio: float = 0.6) -> bool:
    """Reject blocks that are mostly table residue, symbols, or too short to cite."""
    if len(text) < min_chars:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if alpha / max(len(text), 1) < min_alpha_ratio:
        return False
    # Needs at least a handful of real words.
    return len(text.split()) >= 20
