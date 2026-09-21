"""Module 2 (part 2) - text extraction.

Three loaders, one output shape (``Document`` of ``Block``s):

* ``load_jats_xml``    PubMed Central full-text XML - real section hierarchy
* ``load_openfda_json`` FDA structured product labels - fields are sections
* ``load_pdf``          any PDF the user drops into ``data/raw/`` - real pages

XML and JSON have no physical pages, so we synthesise stable pseudo-pages by
packing roughly one printed page of characters at a time. Citations therefore
always resolve to a specific, reproducible location regardless of format.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from lxml import etree

from src.common.logging import get_logger
from src.common.schemas import Block, Document, SourceType
from src.ingestion.cleaner import (
    clean_text,
    detect_running_headers,
    is_meaningful,
    remove_lines,
)

log = get_logger(__name__)

CHARS_PER_SYNTHETIC_PAGE = 2800

# JATS sections that add noise rather than clinical content.
_SKIP_SECTION_KEYWORDS = {
    "acknowledgment",
    "acknowledgments",
    "acknowledgements",
    "author contributions",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "funding",
    "data availability",
    "supplementary material",
    "abbreviations",
    "references",
    "institutional review board statement",
    "informed consent statement",
}

# openFDA label fields worth indexing, mapped to readable section names.
OPENFDA_SECTIONS: dict[str, str] = {
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "boxed_warning": "Boxed Warning",
    "precautions": "Precautions",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "pregnancy": "Pregnancy",
    "nursing_mothers": "Nursing Mothers",
    "pediatric_use": "Pediatric Use",
    "geriatric_use": "Geriatric Use",
    "overdosage": "Overdosage",
    "clinical_pharmacology": "Clinical Pharmacology",
    "mechanism_of_action": "Mechanism of Action",
    "pharmacokinetics": "Pharmacokinetics",
    "pharmacodynamics": "Pharmacodynamics",
    "clinical_studies": "Clinical Studies",
    "description": "Description",
    "information_for_patients": "Information for Patients",
}


def _paginate(
    parts: list[tuple[str, str]], chars_per_page: int = CHARS_PER_SYNTHETIC_PAGE
) -> list[Block]:
    """Pack (section, text) pairs into pseudo-pages of roughly one page each.

    A section never silently spans pages without keeping its name, so every
    resulting block still knows exactly which section it came from.
    """
    blocks: list[Block] = []
    page = 1
    used = 0
    for section, text in parts:
        if not text:
            continue
        # Long sections get split across consecutive pages at paragraph bounds.
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        buffer: list[str] = []
        for para in paragraphs:
            if used + len(para) > chars_per_page and buffer:
                blocks.append(Block(page=page, section=section, text="\n\n".join(buffer)))
                page += 1
                used = 0
                buffer = []
            buffer.append(para)
            used += len(para)
        if buffer:
            blocks.append(Block(page=page, section=section, text="\n\n".join(buffer)))
        # Start the next section on a new page once this one has filled up.
        if used > chars_per_page * 0.66:
            page += 1
            used = 0
    return blocks


# ---------------------------------------------------------------------------
# PubMed Central JATS XML
# ---------------------------------------------------------------------------


def _itertext(el) -> str:
    return " ".join(t.strip() for t in el.itertext() if t and t.strip())


def _section_parts(sec, prefix: str = "") -> list[tuple[str, str]]:
    """Recursively flatten a JATS <sec> into (section path, text) pairs."""
    title = (sec.findtext("title") or "").strip()
    name = f"{prefix} > {title}" if prefix and title else (title or prefix or "Body")

    if any(k in name.lower() for k in _SKIP_SECTION_KEYWORDS):
        return []

    paragraphs = [_itertext(p) for p in sec.findall("./p")]
    body = "\n\n".join(p for p in paragraphs if p)

    parts: list[tuple[str, str]] = []
    if body:
        parts.append((name, body))
    for child in sec.findall("./sec"):
        parts.extend(_section_parts(child, prefix=name))
    return parts


def load_jats_xml(path: Path) -> Document | None:
    try:
        tree = etree.parse(str(path))
    except Exception as exc:
        log.warning("failed to parse %s: %s", path.name, exc)
        return None
    root = tree.getroot()

    title_el = root.find(".//front//article-title")
    title = _itertext(title_el) if title_el is not None else path.stem

    ids = {e.get("pub-id-type"): e.text for e in root.findall(".//front//article-id")}
    pmcid = ids.get("pmcid") or path.stem
    doi = ids.get("doi")

    authors: list[str] = []
    for contrib in root.findall('.//front//contrib[@contrib-type="author"]'):
        surname = contrib.findtext(".//surname")
        given = contrib.findtext(".//given-names")
        if surname:
            authors.append(f"{given} {surname}".strip())

    year = root.findtext('.//front//pub-date/year') or None
    journal = root.findtext(".//front//journal-title")
    lic_el = root.find(".//front//license")
    license_txt = _itertext(lic_el)[:200] if lic_el is not None else None

    parts: list[tuple[str, str]] = []

    abstract = root.find(".//front//abstract")
    if abstract is not None:
        abs_text = "\n\n".join(_itertext(p) for p in abstract.findall(".//p"))
        if abs_text:
            parts.append(("Abstract", abs_text))

    body = root.find(".//body")
    if body is not None:
        for sec in body.findall("./sec"):
            parts.extend(_section_parts(sec))
        # Some articles put paragraphs directly under <body>.
        loose = [_itertext(p) for p in body.findall("./p")]
        loose_text = "\n\n".join(p for p in loose if p)
        if loose_text:
            parts.append(("Body", loose_text))

    cleaned = [(sec, clean_text(txt)) for sec, txt in parts]
    cleaned = [(s, t) for s, t in cleaned if is_meaningful(t, min_chars=80)]
    if not cleaned:
        log.debug("%s produced no usable text", path.name)
        return None

    return Document(
        document_id=pmcid,
        title=title or pmcid,
        source=path.name,
        source_type="research_paper",
        blocks=_paginate(cleaned),
        url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
        authors=authors[:12],
        published=year,
        publisher=journal,
        license=license_txt,
        extra={"doi": doi, "pmid": ids.get("pmid"), "page_model": "synthetic"},
    )


# ---------------------------------------------------------------------------
# openFDA drug labels
# ---------------------------------------------------------------------------


def load_openfda_json(path: Path) -> Document | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("failed to read %s: %s", path.name, exc)
        return None

    label = payload.get("label", {})
    drug = payload.get("drug", path.stem)
    openfda = label.get("openfda", {})

    brand = (openfda.get("brand_name") or [None])[0]
    generic = (openfda.get("generic_name") or [drug])[0]
    title = f"{generic.title()} — FDA Prescribing Information"
    if brand and brand.lower() != generic.lower():
        title = f"{generic.title()} ({brand}) — FDA Prescribing Information"

    parts: list[tuple[str, str]] = []
    for field, section_name in OPENFDA_SECTIONS.items():
        value = label.get(field)
        if not value:
            continue
        text = "\n\n".join(value) if isinstance(value, list) else str(value)
        text = clean_text(text, drop_citations=False)
        if is_meaningful(text, min_chars=80):
            parts.append((section_name, text))

    if not parts:
        return None

    doc_id = f"fda_{generic.lower().replace(' ', '_')}"
    return Document(
        document_id=doc_id,
        title=title,
        source=path.name,
        source_type="drug_label",
        blocks=_paginate(parts),
        url="https://labels.fda.gov/",
        published=(label.get("effective_time") or "")[:4] or None,
        publisher="U.S. Food and Drug Administration",
        license="Public domain (US Government work)",
        extra={
            "generic_name": generic,
            "brand_name": brand,
            "manufacturer": (openfda.get("manufacturer_name") or [None])[0],
            "route": (openfda.get("route") or [None])[0],
            "page_model": "synthetic",
        },
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

# Lines that look like a heading: short, title-ish, no terminal period.
def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not (3 <= len(s) <= 80):
        return False
    if s.endswith((".", ",", ";", ":")) and not s.endswith(":"):
        return False
    words = s.split()
    if len(words) > 10:
        return False
    if s.isupper():
        return True
    # "3.2 Pharmacological Treatment"
    if s[0].isdigit() and any(c.isalpha() for c in s):
        return True
    capitalised = sum(1 for w in words if w[:1].isupper())
    return len(words) >= 2 and capitalised / len(words) >= 0.7


def load_pdf(path: Path, source_type: SourceType = "guideline") -> Document | None:
    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover
        log.error("PyMuPDF not installed; cannot read %s", path.name)
        return None

    try:
        doc = fitz.open(path)
    except Exception as exc:
        log.warning("failed to open %s: %s", path.name, exc)
        return None

    raw_pages = [page.get_text("text") for page in doc]
    meta = doc.metadata or {}
    doc.close()

    headers = detect_running_headers(raw_pages)

    blocks: list[Block] = []
    current_section = "Body"
    for page_no, raw in enumerate(raw_pages, start=1):
        text = remove_lines(raw, headers)
        text = clean_text(text)
        if not text:
            continue

        # Walk the page, promoting heading-like lines to section boundaries.
        buffer: list[str] = []
        for line in text.split("\n"):
            if _looks_like_heading(line):
                if buffer:
                    body = "\n".join(buffer).strip()
                    if is_meaningful(body):
                        blocks.append(
                            Block(page=page_no, section=current_section, text=body)
                        )
                    buffer = []
                current_section = line.strip().rstrip(":")
            else:
                buffer.append(line)
        if buffer:
            body = "\n".join(buffer).strip()
            if is_meaningful(body):
                blocks.append(Block(page=page_no, section=current_section, text=body))

    if not blocks:
        return None

    title = (meta.get("title") or "").strip() or path.stem.replace("_", " ").title()
    return Document(
        document_id=path.stem,
        title=title,
        source=path.name,
        source_type=source_type,
        blocks=blocks,
        authors=[a for a in [(meta.get("author") or "").strip()] if a],
        publisher=(meta.get("producer") or "").strip() or None,
        extra={"page_model": "pdf", "n_pdf_pages": len(raw_pages)},
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def load_document(path: Path) -> Document | None:
    suffix = path.suffix.lower()
    if suffix == ".xml":
        return load_jats_xml(path)
    if suffix == ".json":
        return load_openfda_json(path)
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md"}:
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if not is_meaningful(text):
            return None
        return Document(
            document_id=path.stem,
            title=path.stem.replace("_", " ").title(),
            source=path.name,
            source_type="other",
            blocks=_paginate([("Body", text)]),
            extra={"page_model": "synthetic"},
        )
    return None


SUPPORTED_SUFFIXES = {".xml", ".json", ".pdf", ".txt", ".md"}


def iter_documents(raw_dir: Path) -> Iterator[Document]:
    """Load every supported file under ``raw_dir``, recursively."""
    paths = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in paths:
        try:
            doc = load_document(path)
        except Exception as exc:
            log.warning("error loading %s: %s", path.name, exc)
            continue
        if doc is not None:
            yield doc
