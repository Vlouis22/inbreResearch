"""
Global configuration for the Clinical Text Agent pipeline.
"""

# Model identifiers
CLINICAL_BERT_MODEL = "medicalai/ClinicalBERT"
MEDICAL_T5_MODEL = "Falconsai/medical_summarization"
WHISPER_MODEL = "openai/whisper-small"  # Used for ASR; swap to "base" / "large" as needed

# Summarization parameters
SUMMARIZATION_MAX_LENGTH = 256
SUMMARIZATION_MIN_LENGTH = 64
SUMMARIZATION_MAX_INPUT_TOKENS = 512  # T5 input token limit

# NER confidence threshold (0.0 – 1.0)
NER_CONFIDENCE_THRESHOLD = 0.5

# Entity label groups used for clinical extraction
ENTITY_LABEL_MAP: dict[str, str] = {
    # Symptoms
    "SYMPTOM": "symptoms",
    "SIGN": "symptoms",
    # Conditions / diagnoses
    "DISEASE": "conditions",
    "DISORDER": "conditions",
    "CONDITION": "conditions",
    # Medications
    "DRUG": "medications",
    "MEDICATION": "medications",
    "CHEMICAL": "medications",
    # Procedures
    "PROCEDURE": "procedures",
    "TEST": "procedures",
    # Durations / time
    "DURATION": "durations",
    "DATE": "durations",
    "TIME": "durations",
    # Severity
    "SEVERITY": "severity",
    "DEGREE": "severity",
}
