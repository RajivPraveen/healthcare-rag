"""Cleaning and chunking behaviour that the citation layer depends on."""

from __future__ import annotations

import pytest

from src.common.schemas import Block, Document
from src.common.text import count_tokens, split_sentences, tokenize_words
from src.ingestion.chunker import ChunkConfig, chunk_document, contextual_text
from src.ingestion.cleaner import clean_text, detect_running_headers, is_meaningful


class TestCleaner:
    def test_rejoins_hyphenated_line_wraps(self):
        assert "hypertension" in clean_text("Patients with hyper-\ntension were enrolled.")

    def test_strips_numeric_citations(self):
        cleaned = clean_text("ACE inhibitors reduce mortality [12,14].")
        assert "[12,14]" not in cleaned
        assert "ACE inhibitors reduce mortality" in cleaned

    def test_removes_bare_page_numbers(self):
        assert "42" not in clean_text("Treatment options\n42\nInclude diuretics.")

    def test_preserves_clinical_numbers(self):
        """Dosages and thresholds must survive cleaning."""
        cleaned = clean_text("Start at 10 mg daily; target BP < 130/80 mmHg.")
        assert "10 mg" in cleaned
        assert "130/80" in cleaned

    def test_detects_running_headers(self):
        pages = [f"Journal of Cardiology\nContent for page {i}\nfooter" for i in range(8)]
        assert "Journal of Cardiology" in detect_running_headers(pages)

    def test_is_meaningful_rejects_table_residue(self):
        assert not is_meaningful("| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |")
        assert not is_meaningful("Too short.")
        assert is_meaningful(
            "Angiotensin converting enzyme inhibitors are recommended as initial therapy "
            "for most adults with hypertension, particularly those with diabetes or "
            "chronic kidney disease, because they reduce cardiovascular events."
        )


class TestTokenizer:
    def test_hyphenated_terms_split_consistently(self):
        """'first-line' and 'first line' must produce identical tokens."""
        assert tokenize_words("first-line therapy") == tokenize_words("first line therapy")

    def test_stopwords_removed_but_clinical_terms_kept(self):
        tokens = tokenize_words("What are the risk factors for hypertension?")
        assert "the" not in tokens
        assert "hypertension" in tokens and "risk" in tokens

    def test_sentence_split_ignores_medical_abbreviations(self):
        text = "Give 5 mg b.i.d. as needed. Monitor renal function weekly."
        assert len(split_sentences(text)) == 2


def _document(n_blocks: int = 6) -> Document:
    body = (
        "Angiotensin converting enzyme inhibitors lower blood pressure by blocking "
        "conversion of angiotensin I to angiotensin II. They are recommended as "
        "initial therapy in adults with hypertension and diabetes. Monitor potassium "
        "and creatinine within two weeks of initiation. "
    ) * 4
    return Document(
        document_id="doc_1",
        title="Hypertension Guideline",
        source="doc_1.pdf",
        blocks=[
            Block(page=i + 1, section="Treatment" if i < 3 else "Adverse Effects", text=body)
            for i in range(n_blocks)
        ],
    )


class TestChunker:
    def test_respects_target_size(self):
        chunks = chunk_document(_document(), ChunkConfig(target_tokens=200, overlap_tokens=30))
        assert chunks
        # Allow one sentence of spill past the target.
        assert all(c.token_count <= 200 + 80 for c in chunks)

    def test_never_crosses_section_boundaries(self):
        """A chunk spanning sections would produce a miscited clinical claim."""
        chunks = chunk_document(_document(), ChunkConfig(target_tokens=4000, overlap_tokens=0))
        assert {c.section for c in chunks} == {"Treatment", "Adverse Effects"}

    def test_carries_citation_metadata(self):
        chunk = chunk_document(_document())[0]
        assert chunk.document_id == "doc_1"
        assert chunk.page >= 1
        assert chunk.section
        assert chunk.title == "Hypertension Guideline"

    def test_chunk_ids_are_unique(self):
        chunks = chunk_document(_document(), ChunkConfig(target_tokens=150, overlap_tokens=20))
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_overlap_creates_shared_text(self):
        chunks = chunk_document(_document(2), ChunkConfig(target_tokens=120, overlap_tokens=40))
        assert len(chunks) >= 2
        first_words = set(chunks[0].text.split())
        assert first_words & set(chunks[1].text.split())

    @pytest.mark.parametrize("target", [300, 500, 800, 1200])
    def test_sweep_sizes_all_produce_chunks(self, target):
        chunks = chunk_document(
            _document(10), ChunkConfig(target_tokens=target, overlap_tokens=int(target * 0.15))
        )
        assert chunks
        assert all(c.token_count > 0 for c in chunks)

    def test_contextual_text_prefixes_provenance(self):
        chunk = chunk_document(_document())[0]
        prefixed = contextual_text(chunk)
        assert prefixed.startswith("Hypertension Guideline")
        assert chunk.section in prefixed
        assert count_tokens(prefixed) > count_tokens(chunk.text) - 1
