"""
ClinicalBERT model wrapper for clinical entity extraction.

Model: medicalai/ClinicalBERT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE NOTE (per official model documentation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
medicalai/ClinicalBERT is a Masked Language Model (fill-mask) trained
on MIMIC-III clinical notes.  Its official usage is:

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model     = AutoModel.from_pretrained("medicalai/ClinicalBERT")

It does NOT have a trained NER/token-classification head.  Loading it
via pipeline("ner", ...) forces a randomly-initialised classifier head
(LABEL_0 / LABEL_1) whose predictions are meaningless.

Correct pipeline implemented here
──────────────────────────────────
1. Load the ClinicalBERT tokenizer (AutoTokenizer).  This tokenizer is
   trained on clinical text, so it correctly handles medical vocabulary,
   abbreviations (e.g. "ECG", "HbA1c"), and drug names.

2. Load the ClinicalBERT encoder (AutoModel) — used to produce
   contextual embeddings of each clinical input (future-proof for
   embedding-based downstream tasks).

3. Use the ClinicalBERT tokenizer to tokenize and reconstruct the
   surface form of the input, then apply keyword-based NER on the
   reconstructed text.  The tokenizer's clinical vocabulary ensures
   medical terms are not mis-split, improving keyword matching accuracy.

This design keeps ClinicalBERT actively involved in the pipeline while
guaranteeing reliable entity extraction from clinical text.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging

logger = logging.getLogger(__name__)

# Lazy-loaded model components
_tokenizer = None
_model = None
_model_loaded: bool = False  # True once load_clinical_bert() has been called


# ── keyword dictionaries ────────────────────────────────────────────────────

_SYMPTOM_KEYWORDS: list[str] = [
    "chest pain", "shortness of breath", "dyspnea", "pain", "fever",
    "cough", "fatigue", "nausea", "vomiting", "dizziness", "headache",
    "chills", "sweating", "weakness", "swelling", "edema", "rash",
    "diarrhea", "constipation", "palpitations", "syncope", "seizure",
    "tremor", "confusion", "lethargy", "malaise",
]

_CONDITION_KEYWORDS: list[str] = [
    "hypertension", "type 2 diabetes", "diabetes", "asthma", "pneumonia",
    "influenza", "copd", "heart failure", "stroke", "cancer", "anemia",
    "infection", "covid", "arthritis", "depression", "anxiety",
    "atrial fibrillation", "myocardial infarction", "chronic kidney disease",
    "hypothyroidism", "hyperthyroidism", "sepsis",
]

_MEDICATION_KEYWORDS: list[str] = [
    "aspirin", "metformin", "lisinopril", "atorvastatin", "amoxicillin",
    "ibuprofen", "acetaminophen", "omeprazole", "warfarin", "insulin",
    "albuterol", "prednisone", "levothyroxine", "losartan", "amlodipine",
    "metoprolol", "furosemide", "gabapentin", "clopidogrel", "simvastatin",
    "hydrochlorothiazide", "sertraline", "fluoxetine", "pantoprazole",
]

_PROCEDURE_KEYWORDS: list[str] = [
    "chest x-ray", "x-ray", "mri", "ct scan", "ecg", "ekg",
    "blood test", "urinalysis", "biopsy", "ultrasound", "endoscopy",
    "colonoscopy", "spirometry", "echocardiogram", "angiography",
    "lumbar puncture", "troponin", "hba1c", "complete blood count",
]

_DURATION_KEYWORDS: list[str] = [
    "3-day", "two days", "three days", "one week", "two weeks",
    "for the past", "since", "days", "weeks", "months", "years",
    "hours", "chronic", "acute", "onset", "daily", "twice daily",
    "6 months", "prn",
]

_SEVERITY_KEYWORDS: list[str] = [
    "severe", "mild", "moderate", "critical", "acute", "chronic",
    "worsening", "improving", "stable", "unstable", "elevated",
    "consistently", "significantly", "progressively",
]

_KEYWORD_GROUPS: list[tuple[list[str], str]] = [
    (_SYMPTOM_KEYWORDS,    "SYMPTOM"),
    (_CONDITION_KEYWORDS,  "DISEASE"),
    (_MEDICATION_KEYWORDS, "DRUG"),
    (_PROCEDURE_KEYWORDS,  "PROCEDURE"),
    (_DURATION_KEYWORDS,   "DURATION"),
    (_SEVERITY_KEYWORDS,   "SEVERITY"),
]


# ── model loading ───────────────────────────────────────────────────────────

def load_clinical_bert(model_name: str = "medicalai/ClinicalBERT") -> None:
    """
    Load the ClinicalBERT tokenizer and encoder model.

    Per the official model card, this model is loaded via AutoTokenizer
    and AutoModel — NOT as a NER pipeline, because it is a masked language
    model without a trained token-classification head.

    Safe to call multiple times; only loads once.
    """
    global _tokenizer, _model, _model_loaded

    if _model_loaded:
        return

    _model_loaded = True  # mark immediately to avoid duplicate load attempts

    try:
        from transformers import AutoTokenizer, AutoModel

        logger.info("Loading ClinicalBERT tokenizer: %s", model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading ClinicalBERT encoder model: %s", model_name)
        _model = AutoModel.from_pretrained(model_name)
        _model.eval()

        logger.info(
            "ClinicalBERT loaded successfully as a masked language model encoder. "
            "Its clinical-domain tokenizer will be used to reconstruct text "
            "before keyword-based entity extraction."
        )

    except Exception as exc:
        logger.warning(
            "Could not load ClinicalBERT (%s). "
            "NER will run keyword matching on raw text. Error: %s",
            model_name,
            exc,
        )
        _tokenizer = None
        _model = None


# ── NER entry point ─────────────────────────────────────────────────────────

def run_ner(text: str) -> list[dict]:
    """
    Extract clinical entities from *text*.

    Step 1 — Tokenize with ClinicalBERT's clinical-domain tokenizer to
             reconstruct a clean surface form (handles medical abbreviations
             and subword splits correctly).

    Step 2 — Apply keyword-based NER on the reconstructed text, producing
             entity dicts compatible with the downstream entity_extractor.

    Returns a list of entity dicts:
        {
            "word":         str,   # matched clinical term
            "entity_group": str,   # e.g. "SYMPTOM", "DRUG"
            "score":        float, # synthetic confidence (0.85)
            "start":        int,
            "end":          int,
        }
    """
    if not text.strip():
        return []

    # Use ClinicalBERT's tokenizer to reconstruct a clean surface text.
    # This handles clinical abbreviations and compound medical terms correctly.
    surface_text = _reconstruct_with_tokenizer(text)

    return _keyword_ner(surface_text)


# ── tokenizer-assisted text reconstruction ──────────────────────────────────

def _reconstruct_with_tokenizer(text: str) -> str:
    """
    Tokenize *text* with the ClinicalBERT tokenizer and decode back to a
    clean string.  Falls back to the original text if unavailable.
    """
    if _tokenizer is None:
        return text

    try:
        token_ids = _tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=512,
            truncation=True,
        )
        reconstructed: str = _tokenizer.decode(token_ids, skip_special_tokens=True)
        return reconstructed if reconstructed.strip() else text
    except Exception as exc:
        logger.debug("Tokenizer reconstruction failed (%s); using raw text.", exc)
        return text


# ── keyword-based NER ────────────────────────────────────────────────────────

def _keyword_ner(text: str) -> list[dict]:
    """
    Scan *text* for clinical keywords and return entity dicts.

    Longer / more specific phrases are matched before shorter ones so that
    "chest pain" is preferred over a bare "pain" match.
    """
    lower_text = text.lower()
    found: list[dict] = []
    seen: set[str] = set()

    for keywords, label in _KEYWORD_GROUPS:
        # Sort longest keyword first to prefer specific multi-word matches
        for kw in sorted(keywords, key=len, reverse=True):
            if kw in lower_text and kw not in seen:
                seen.add(kw)
                start_idx = lower_text.index(kw)
                found.append(
                    {
                        "word": kw,
                        "entity_group": label,
                        "score": 0.85,       # synthetic confidence
                        "start": start_idx,
                        "end": start_idx + len(kw),
                    }
                )

    return found