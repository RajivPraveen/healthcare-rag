"""Module 1 - Data collection.

Pulls a real healthcare corpus from two public, license-clean sources:

* PubMed Central Open Access subset (clinical reviews / guideline papers) via E-utilities
* openFDA drug label database (structured product labelling)

Everything lands in ``data/raw/`` alongside a manifest in ``data/metadata/``.
Downloads are idempotent: files already on disk are skipped.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from src.common.config import get_settings
from src.common.logging import get_logger

log = get_logger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENFDA_LABEL = "https://api.fda.gov/drug/label.json"

# NCBI asks for <=3 req/s without an API key.
_NCBI_DELAY = 0.4

# Searched as `<concept>[Title]`. Constraining to the title is what keeps the
# corpus clinically on-topic: relevance-sorted free-text search happily returns
# epidemiology papers that merely mention the condition in passing.
CLINICAL_TOPICS: list[str] = [
    "hypertension",
    "type 2 diabetes",
    "asthma",
    "chronic obstructive pulmonary disease",
    "heart failure",
    "atrial fibrillation",
    "ischemic stroke",
    "dyslipidemia",
    "major depressive disorder",
    "anxiety disorder",
    "sepsis",
    "pneumonia",
    "chronic kidney disease",
    "obesity",
    "osteoporosis",
    "migraine",
    "COVID-19",
    "antimicrobial stewardship",
    "venous thromboembolism",
    "hypothyroidism",
    "diabetic ketoacidosis",
    "acute coronary syndrome",
    "atopic dermatitis",
    "rheumatoid arthritis",
    "inflammatory bowel disease",
    "chronic pain",
    "anticoagulation",
    "urinary tract infection",
    "gastroesophageal reflux disease",
    "obstructive sleep apnea",
]

COMMON_DRUGS: list[str] = [
    "lisinopril",
    "amlodipine",
    "metformin",
    "atorvastatin",
    "losartan",
    "metoprolol",
    "albuterol",
    "omeprazole",
    "sertraline",
    "gabapentin",
    "levothyroxine",
    "hydrochlorothiazide",
    "warfarin",
    "apixaban",
    "prednisone",
    "amoxicillin",
    "ibuprofen",
    "furosemide",
    "insulin glargine",
    "montelukast",
    "escitalopram",
    "clopidogrel",
    "rosuvastatin",
    "pantoprazole",
    "tamsulosin",
]


@dataclass
class DownloadReport:
    pmc: int = 0
    openfda: int = 0
    skipped: int = 0
    failed: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.pmc + self.openfda


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(45.0),
        follow_redirects=True,
        headers={"User-Agent": "healthcare-rag/0.1 (research project; contact: local)"},
    )


# ---------------------------------------------------------------------------
# PubMed Central Open Access
# ---------------------------------------------------------------------------


def _pmc_search(client: httpx.Client, topic: str, retmax: int) -> list[str]:
    """Return PMC IDs for a topic, restricted to the Open Access subset.

    The topic must appear in the *title*, and the article must be a review
    discussing treatment/management. Without the title constraint the results
    drift into papers that only mention the condition as a comorbidity.
    """
    term = (
        f"{topic}[Title] "
        "AND (treatment[Title/Abstract] OR management[Title/Abstract] "
        "OR therapy[Title/Abstract] OR guideline[Title/Abstract]) "
        'AND "open access"[filter] AND review[Publication Type]'
    )
    resp = client.get(
        f"{EUTILS}/esearch.fcgi",
        params={
            "db": "pmc",
            "term": term,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
        },
    )
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def _pmc_fetch(client: httpx.Client, pmcid: str) -> bytes | None:
    resp = client.get(
        f"{EUTILS}/efetch.fcgi", params={"db": "pmc", "id": pmcid, "retmode": "xml"}
    )
    resp.raise_for_status()
    content = resp.content
    # Articles outside the OA subset come back as a stub with no <body>.
    if b"<body" not in content:
        return None
    return content


def download_pmc(
    out_dir: Path,
    topics: list[str] | None = None,
    per_topic: int = 4,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    topics = topics or CLINICAL_TOPICS
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    with _client() as client:
        for topic in topics:
            if limit and len(records) >= limit:
                break
            try:
                ids = _pmc_search(client, topic, per_topic)
            except Exception as exc:  # network hiccup on one topic shouldn't kill the run
                log.warning("PMC search failed for %r: %s", topic, exc)
                continue
            time.sleep(_NCBI_DELAY)

            for pmcid in ids:
                if limit and len(records) >= limit:
                    break
                if pmcid in seen:
                    continue
                seen.add(pmcid)

                path = out_dir / f"PMC{pmcid}.xml"
                if path.exists() and path.stat().st_size > 0:
                    log.debug("skip existing %s", path.name)
                    records.append(
                        {"path": str(path), "source_type": "research_paper", "topic": topic}
                    )
                    continue

                try:
                    content = _pmc_fetch(client, pmcid)
                    time.sleep(_NCBI_DELAY)
                except Exception as exc:
                    log.warning("PMC fetch failed for %s: %s", pmcid, exc)
                    continue

                if content is None:
                    log.debug("PMC%s has no full text in the OA subset, skipping", pmcid)
                    continue

                path.write_bytes(content)
                log.info("PMC%s saved (%s, %.0f KB)", pmcid, topic, len(content) / 1024)
                records.append(
                    {"path": str(path), "source_type": "research_paper", "topic": topic}
                )

    return records


# ---------------------------------------------------------------------------
# openFDA drug labels
# ---------------------------------------------------------------------------


def download_openfda(
    out_dir: Path, drugs: list[str] | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    drugs = drugs or COMMON_DRUGS
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with _client() as client:
        for drug in drugs:
            if limit and len(records) >= limit:
                break
            slug = drug.replace(" ", "_").lower()
            path = out_dir / f"fda_{slug}.json"
            if path.exists() and path.stat().st_size > 0:
                records.append({"path": str(path), "source_type": "drug_label", "topic": drug})
                continue

            try:
                resp = client.get(
                    OPENFDA_LABEL,
                    params={
                        "search": f'openfda.generic_name:"{drug}"',
                        "limit": 1,
                    },
                )
                if resp.status_code == 404:
                    log.debug("no openFDA label for %s", drug)
                    continue
                resp.raise_for_status()
                results = resp.json().get("results", [])
            except Exception as exc:
                log.warning("openFDA fetch failed for %s: %s", drug, exc)
                continue

            if not results:
                continue

            payload = {"drug": drug, "label": results[0]}
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info("openFDA label saved for %s", drug)
            records.append({"path": str(path), "source_type": "drug_label", "topic": drug})
            time.sleep(0.2)

    return records


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_corpus(
    pmc_per_topic: int = 4,
    pmc_limit: int | None = None,
    fda_limit: int | None = None,
    topics: list[str] | None = None,
    drugs: list[str] | None = None,
) -> DownloadReport:
    """Download the full corpus and write ``data/metadata/corpus_manifest.json``."""
    settings = get_settings()
    report = DownloadReport()

    log.info("Downloading PubMed Central Open Access articles…")
    pmc_records = download_pmc(
        settings.raw_dir / "pmc", topics=topics, per_topic=pmc_per_topic, limit=pmc_limit
    )
    report.pmc = len(pmc_records)

    log.info("Downloading openFDA drug labels…")
    fda_records = download_openfda(settings.raw_dir / "openfda", drugs=drugs, limit=fda_limit)
    report.openfda = len(fda_records)

    report.records = pmc_records + fda_records

    manifest = settings.metadata_dir / "corpus_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "counts": {
                    "pmc": report.pmc,
                    "openfda": report.openfda,
                    "total": report.total,
                },
                "records": report.records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Corpus ready: %d documents (manifest: %s)", report.total, manifest)
    return report
