"""
ClinicalBERT model wrapper for clinical entity extraction.

Model: medicalai/ClinicalBERT

ARCHITECTURE NOTE (per official model documentation)
─────────────────────────────────────────────────────
medicalai/ClinicalBERT is a Masked Language Model (fill-mask) trained
on MIMIC-III clinical notes.  Official usage:

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model     = AutoModel.from_pretrained("medicalai/ClinicalBERT")

It has NO trained NER/token-classification head.  Loading it as a NER
pipeline produces a randomly-initialised head (LABEL_0/LABEL_1) whose
predictions are meaningless.

Pipeline used here
──────────────────
1.  Load AutoTokenizer + AutoModel (per model card).
2.  Tokenize → decode with the ClinicalBERT tokenizer so that clinical
    abbreviations and compound terms are handled correctly.
3.  Apply keyword-based NER on the reconstructed text.
4.  Return entity dicts compatible with entity_extractor downstream.

Exported constants (used by entity_fusion for post-processing)
──────────────────────────────────────────────────────────────
ABBREVIATION_MAP   str → canonical form  ("ecg" → "ECG")
SUBSUMPTION_MAP    broad term → set of specific phrases that subsume it
                   e.g. "pain" → {"chest pain", "abdominal pain", ...}
                   If any specific phrase is present, the broad term is dropped.
"""

import logging

logger = logging.getLogger(__name__)

# ── lazy-loaded model components ────────────────────────────────────────────
_tokenizer = None
_model = None
_model_loaded: bool = False


# ── abbreviation normalisation ───────────────────────────────────────────────
# Maps lowercase surface form → preferred display string.
ABBREVIATION_MAP: dict[str, str] = {
    "ecg":   "ECG",
    "ekg":   "ECG",          # normalise EKG → ECG
    "mri":   "MRI",
    "ct scan": "CT scan",
    "hba1c": "HbA1c",
    "copd":  "COPD",
    "prn":   "as needed",
    "3-day": "3 days",
    "covid": "COVID-19",
    "ekg":   "ECG",
    "mrsa":  "MRSA",
    "dvt":   "DVT",
    "pe":    "PE",
    "uti":   "UTI",
    "uri":   "URI",
}

# ── subsumption map ──────────────────────────────────────────────────────────
# broad_term → set of more-specific phrases that make the broad term redundant.
# Applies within the SAME entity category.
SUBSUMPTION_MAP: dict[str, set[str]] = {
    # symptoms
    "pain":          {"chest pain", "abdominal pain", "back pain", "joint pain",
                      "throat pain", "flank pain", "pelvic pain"},
    "breath":        {"shortness of breath"},
    "swelling":      {"edema", "peripheral edema"},
    # conditions
    "diabetes":      {"type 2 diabetes", "type 1 diabetes",
                      "gestational diabetes", "diabetic"},
    "kidney disease": {"chronic kidney disease"},
    # procedures
    "x-ray":         {"chest x-ray"},
}


# ── keyword dictionaries ─────────────────────────────────────────────────────
# Longer / more specific phrases must appear BEFORE shorter ones in each list
# because _keyword_ner() sorts by length descending before scanning.

_SYMPTOM_KEYWORDS: list[str] = [
    "shortness of breath", "chest pain", "abdominal pain", "back pain",
    "joint pain", "throat pain", "peripheral edema",
    "dyspnea", "fever", "cough", "fatigue", "nausea", "vomiting",
    "dizziness", "headache", "chills", "sweating", "weakness",
    "edema", "rash", "diarrhea", "constipation", "palpitations",
    "syncope", "seizure", "tremor", "confusion", "lethargy",
    "malaise", "pain",
]

_CONDITION_KEYWORDS: list[str] = [
    "type 2 diabetes", "type 1 diabetes", "chronic kidney disease",
    "atrial fibrillation", "myocardial infarction", "heart failure",
    "hypertension", "pneumonia", "influenza", "asthma",
    "copd", "stroke", "cancer", "anemia", "infection",
    "covid", "arthritis", "depression", "anxiety",
    "hypothyroidism", "hyperthyroidism", "sepsis",
    "diabetes", "hyperlipidemia", "obesity",
]

_MEDICATION_KEYWORDS: list[str] = [
    "hydrochlorothiazide", "levothyroxine", "atorvastatin",
    "amoxicillin", "acetaminophen", "simvastatin",
    "pantoprazole", "clopidogrel", "metoprolol",
    "furosemide", "gabapentin", "sertraline",
    "fluoxetine", "amlodipine", "omeprazole",
    "lisinopril", "metformin", "warfarin",
    "losartan", "prednisone", "albuterol",
    "ibuprofen", "insulin", "aspirin",
]

_PROCEDURE_KEYWORDS: list[str] = [
    "complete blood count", "chest x-ray", "lumbar puncture",
    "echocardiogram", "angiography", "colonoscopy",
    "endoscopy", "urinalysis", "blood test",
    "ultrasound", "spirometry", "biopsy",
    "troponin", "hba1c", "ct scan",
    "ekg", "ecg", "mri",
    "x-ray",
]

_DURATION_KEYWORDS: list[str] = [
    "for the past three days", "for the past two days", "for the past week",
    "for the past month", "for the past year",
    "three days", "two days", "one week", "two weeks",
    "3-day", "6 months", "twice daily",
    "days", "weeks", "months", "years", "hours",
    "chronic", "acute", "onset", "daily", "prn",
    "for the past", "since",
]

_SEVERITY_KEYWORDS: list[str] = [
    "progressively", "significantly", "consistently",
    "worsening", "improving", "unstable",
    "critical", "elevated", "moderate",
    "severe", "stable", "chronic",
    "acute", "mild",
]

_KEYWORD_GROUPS: list[tuple[list[str], str]] = [
    (_SYMPTOM_KEYWORDS,    "SYMPTOM"),
    (_CONDITION_KEYWORDS,  "DISEASE"),
    (_MEDICATION_KEYWORDS, "DRUG"),
    (_PROCEDURE_KEYWORDS,  "PROCEDURE"),
    (_DURATION_KEYWORDS,   "DURATION"),
    (_SEVERITY_KEYWORDS,   "SEVERITY"),
]


# model loading

def load_clinical_bert(model_name: str = "medicalai/ClinicalBERT") -> None:
    """
    Load the ClinicalBERT tokenizer and encoder (AutoTokenizer + AutoModel).
    Per the official model card — NOT as a NER pipeline.
    Safe to call multiple times; only loads once.
    """
    global _tokenizer, _model, _model_loaded
    if _model_loaded:
        return
    _model_loaded = True

    try:
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading ClinicalBERT tokenizer: %s", model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Loading ClinicalBERT encoder: %s", model_name)
        _model = AutoModel.from_pretrained(model_name)
        _model.eval()
        logger.info(
            "ClinicalBERT loaded. Clinical-domain tokenizer active for "
            "text reconstruction before entity extraction."
        )
    except Exception as exc:
        logger.warning(
            "Could not load ClinicalBERT (%s). "
            "Keyword NER will run on raw text. Error: %s",
            model_name, exc,
        )
        _tokenizer = None
        _model = None


# NER entry point 

def run_ner(text: str) -> list[dict]:
    """
    Extract clinical entities from *text*.

    1. Tokenize → decode with ClinicalBERT tokenizer (clinical vocabulary).
    2. Keyword-based NER on the reconstructed surface text.

    Returns entity dicts with keys: word, entity_group, score, start, end.
    """
    if not text.strip():
        return []
    surface = _reconstruct_with_tokenizer(text)
    return _keyword_ner(surface)


# tokenizer reconstruction 

def _reconstruct_with_tokenizer(text: str) -> str:
    """Round-trip through the ClinicalBERT tokenizer to normalise the text."""
    if _tokenizer is None:
        return text
    try:
        ids = _tokenizer.encode(
            text, add_special_tokens=False, max_length=512, truncation=True,
        )
        decoded = _tokenizer.decode(ids, skip_special_tokens=True)
        return decoded if decoded.strip() else text
    except Exception as exc:
        logger.debug("Tokenizer round-trip failed (%s); using raw text.", exc)
        return text


# keyword NER

def _keyword_ner(text: str) -> list[dict]:
    """
    Scan *text* for clinical keywords longest-first (greedy match).
    Returns entity dicts compatible with entity_extractor.
    """
    lower = text.lower()
    found: list[dict] = []
    # Track matched character spans to avoid sub-matches inside longer matches
    matched_spans: list[tuple[int, int]] = []
    seen_words: set[str] = set()

    for keywords, label in _KEYWORD_GROUPS:
        for kw in sorted(keywords, key=len, reverse=True):
            idx = lower.find(kw)
            while idx != -1:
                end = idx + len(kw)
                # Skip if this span is inside an already-matched longer span
                overlaps = any(s <= idx and end <= e for s, e in matched_spans)
                if not overlaps and kw not in seen_words:
                    seen_words.add(kw)
                    matched_spans.append((idx, end))
                    found.append({
                        "word": kw,
                        "entity_group": label,
                        "score": 0.85,
                        "start": idx,
                        "end": end,
                    })
                idx = lower.find(kw, idx + 1)

    return found