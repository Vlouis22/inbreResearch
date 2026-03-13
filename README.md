# Clinical Text Agent

A modular Python pipeline that converts multi-source clinical inputs into a **Structured Patient Profile**.  
The pipeline stops at the Structured Patient Profile stage — diagnostic reasoning is handled by a separate downstream system.

---

## Architecture

```
Clinical Inputs
   ├── Doctor Notes      (text)
   ├── Patient Voice     (audio) ──► Speech-to-Text (ASR)
   └── Health Records    (text)
            │
            ▼
   Text Normalization
            │
     ┌──────┴──────┐
     ▼             ▼
ClinicalBERT   Medical T5
 (NER / entity  (summarization)
  extraction)
     │             │
 Entities     Medical Summary
 Input 1..N       │
     │             │
 Entity Fusion     │
     │             │
     └──────┬──────┘
            ▼
  Structured Patient Profile
  ┌─────────────────────────────┐
  │  clinical_entities  (JSON)  │
  │  medical_summary   (string) │
  │  source_metadata   (flags)  │
  └─────────────────────────────┘
```

---

## Project Structure

```
clinical_text_agent/
├── main.py                          # Entry point & pipeline orchestrator
├── config.py                        # Model names, thresholds, label maps
├── requirements.txt
├── models/
│   ├── asr_model.py                 # Whisper ASR (+ placeholder fallback)
│   ├── clinical_bert_model.py       # ClinicalBERT NER (+ keyword fallback)
│   └── medical_t5_model.py          # Medical T5 summarization (+ extractive fallback)
├── pipeline/
│   ├── text_normalizer.py           # Text cleaning & normalization
│   ├── entity_extractor.py          # Map NER output → ClinicalEntities
│   ├── entity_fusion.py             # Merge & deduplicate entities across sources
│   └── summarizer.py                # Combine text & run T5 summarization
├── schemas/
│   ├── clinical_entities.py         # Pydantic schema for extracted entities
│   └── structured_patient_profile.py# Pydantic schema for final output
└── utils/
    └── text_utils.py                # Shared text helper functions
```

---

## Models Used

| Role | Model |
|------|-------|
| Clinical NER | [`medicalai/ClinicalBERT`](https://huggingface.co/medicalai/ClinicalBERT) |
| Medical summarization | [`Falconsai/medical_summarization`](https://huggingface.co/Falconsai/medical_summarization) |
| Speech-to-text (ASR) | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) |

All models include **graceful fallbacks** so the pipeline runs end-to-end even without internet access or a GPU.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

Example output:

```json
{
  "clinical_entities": {
    "symptoms": ["chest pain", "shortness of breath"],
    "conditions": ["hypertension", "type 2 diabetes", "pneumonia", "COPD"],
    "medications": ["metformin", "lisinopril", "albuterol"],
    "procedures": ["ECG", "chest X-ray", "MRI brain", "spirometry"],
    "durations": ["3-day", "6 months ago"],
    "severity": ["severe", "mild", "moderate"]
  },
  "medical_summary": "58-year-old male with hypertension and diabetes presenting with chest pain and shortness of breath ...",
  "source_metadata": {
    "doctor_notes": true,
    "patient_conversation": true,
    "health_records": true
  }
}
```

---

## Input Types

Each `ClinicalInput` has a `source_type` and `content`:

```python
ClinicalInput(source_type="doctor_notes",        content="Patient presents with ...")
ClinicalInput(source_type="patient_conversation", content="recording.wav")   # audio → ASR
ClinicalInput(source_type="health_records",       content="Previous admission ...")
```

Audio files are detected by extension (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`).

---

## Extending

- **Add a new entity category**: Update `ENTITY_LABEL_MAP` in `config.py` and add the field to `ClinicalEntities`.
- **Swap the ASR model**: Change `WHISPER_MODEL` in `config.py`.
- **Swap the summarization model**: Change `MEDICAL_T5_MODEL` in `config.py`.
