"""Query-side preprocessing.

Clinical questions are written in shorthand while the literature is written in
full ("MI" vs "myocardial infarction", "HTN" vs "hypertension"). Dense
retrieval partially absorbs this; BM25 cannot, since the two forms share no
tokens. Expanding the query — rather than the 3.5k indexed chunks — keeps the
index untouched and costs nothing at build time.
"""

from __future__ import annotations

import re

# Deliberately curated rather than exhaustive: a wrong expansion is worse than
# a missing one, because it injects a false term into the lexical query.
MEDICAL_ABBREVIATIONS: dict[str, str] = {
    "htn": "hypertension",
    "bp": "blood pressure",
    "mi": "myocardial infarction",
    "chf": "congestive heart failure",
    "hf": "heart failure",
    "hfref": "heart failure reduced ejection fraction",
    "cad": "coronary artery disease",
    "acs": "acute coronary syndrome",
    "af": "atrial fibrillation",
    "afib": "atrial fibrillation",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "vte": "venous thromboembolism",
    "cva": "cerebrovascular accident stroke",
    "tia": "transient ischemic attack",
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "dm": "diabetes mellitus",
    "dka": "diabetic ketoacidosis",
    "hba1c": "hemoglobin a1c glycated hemoglobin",
    "ckd": "chronic kidney disease",
    "aki": "acute kidney injury",
    "esrd": "end stage renal disease",
    "egfr": "estimated glomerular filtration rate",
    "copd": "chronic obstructive pulmonary disease",
    "ards": "acute respiratory distress syndrome",
    "osa": "obstructive sleep apnea",
    "uti": "urinary tract infection",
    "cap": "community acquired pneumonia",
    "gerd": "gastroesophageal reflux disease",
    "ibd": "inflammatory bowel disease",
    "ra": "rheumatoid arthritis",
    "sle": "systemic lupus erythematosus",
    "mdd": "major depressive disorder",
    "gad": "generalized anxiety disorder",
    "ssri": "selective serotonin reuptake inhibitor",
    "snri": "serotonin norepinephrine reuptake inhibitor",
    "ace": "angiotensin converting enzyme",
    "acei": "angiotensin converting enzyme inhibitor",
    "arb": "angiotensin receptor blocker",
    "ccb": "calcium channel blocker",
    "nsaid": "nonsteroidal anti-inflammatory drug",
    "ppi": "proton pump inhibitor",
    "doac": "direct oral anticoagulant",
    "ldl": "low density lipoprotein cholesterol",
    "hdl": "high density lipoprotein cholesterol",
    "bmi": "body mass index",
    "icu": "intensive care unit",
    "ed": "emergency department",
    "iv": "intravenous",
    "po": "oral administration",
    "prn": "as needed",
    "bid": "twice daily",
    "tid": "three times daily",
    "qd": "once daily",
    "rct": "randomized controlled trial",
    "nnt": "number needed to treat",
    "ci": "confidence interval",
}

_WORD = re.compile(r"\b[\w-]+\b")


def expand_query(query: str) -> str:
    """Append expansions for any recognised abbreviations, keeping the original.

    The original token is retained so exact matches still score; the expansion
    is additive rather than a substitution.
    """
    additions: list[str] = []
    seen: set[str] = set()
    for match in _WORD.finditer(query):
        token = match.group(0).lower()
        expansion = MEDICAL_ABBREVIATIONS.get(token)
        if expansion and expansion not in seen:
            additions.append(expansion)
            seen.add(expansion)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


_QUESTION_PREFIXES = re.compile(
    r"^\s*(?:can you |could you |please |tell me |i want to know |what is |what are |"
    r"how do i |how does |explain )",
    re.I,
)


def normalize_question(question: str) -> str:
    """Light normalisation used for cache keys and logging, not for retrieval."""
    return " ".join(question.strip().split())


def looks_out_of_scope(question: str) -> bool:
    """Cheap guard for obviously non-clinical input before spending a retrieval."""
    text = question.strip()
    return len(text) < 3 or not any(c.isalpha() for c in text)
