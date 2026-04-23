# Dr. House Extraction Pipelines

A modular Python pipeline that converts multi-source clinical inputs and lab inputs into a **Structured Patient Profile**.  
The pipeline stops at the Structured Patient Profile stage — diagnostic reasoning is handled by a separate downstream system.

---

## Architecture

```
Clinical Inputs                    Lab Inputs
   ├── Doctor Notes      (text)
   ├── Patient Voice     (audio) ──► Speech-to-Text (ASR)
   └── Health Records    (text)     ├── Image / PDF report
            │                       ├── Digital table
            ▼                       └── OCR text / inline report
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
        Lab Agent
  ┌─────────────────────────────┐
  │  row extraction             │
  │  unit / test normalization  │
  │  abnormality engine         │
  └─────────────────────────────┘
            │
            ▼
  Structured Patient Profile
  ┌─────────────────────────────────────────┐
  │  clinical_entities  (JSON)              │
  │  medical_summary   (string)             │
  │  lab_profile       (structured JSON)    │
  │  source_metadata   (flags)              │
  └─────────────────────────────────────────┘
```

---

## Project Structure

```
inbreResearch/
├── main.py                          # Entry point & pipeline orchestrator
├── lab_main.py                      # Standalone Lab Agent demo
├── lab_eval_cases.py                # Synthetic gold lab evaluation set
├── config.py                        # Model names, thresholds, label maps
├── requirements.txt
├── models/
│   ├── asr_model.py                 # Whisper ASR (+ placeholder fallback)
│   ├── clinical_bert_model.py       # ClinicalBERT NER (+ keyword fallback)
│   ├── lab_document_model.py        # HF OCR wrapper (+ local fallbacks)
│   └── medical_t5_model.py          # Medical T5 summarization (+ extractive fallback)
├── pipeline/
│   ├── text_normalizer.py           # Text cleaning & normalization
│   ├── entity_extractor.py          # Map NER output → ClinicalEntities
│   ├── entity_fusion.py             # Merge & deduplicate entities across sources
│   ├── summarizer.py                # Combine text & run T5 summarization
│   ├── lab_pipeline.py              # Lab Agent orchestration
│   ├── lab_row_parser.py            # Structured lab row extraction
│   ├── lab_normalizer.py            # Test-name / unit normalization
│   └── lab_abnormality.py           # Deterministic abnormality engine
├── schemas/
│   ├── clinical_entities.py         # Pydantic schema for extracted entities
│   ├── lab_profile.py               # Pydantic schema for lab outputs
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
| Lab OCR / document parsing | [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) |
| Lab OCR fallback | [`nanonets/Nanonets-OCR2-3B`](https://huggingface.co/nanonets/Nanonets-OCR2-3B) |

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
python lab_main.py
python test_lab_pipeline.py
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
  "lab_profile": {
    "lab_results": [
      {
        "test_name_canonical": "White Blood Cell Count",
        "value_numeric_optional": 12.4,
        "unit_canonical": "10^3/uL",
        "computed_flag": "high"
      }
    ],
    "abnormal_findings": ["leukocytosis", "low hemoglobin"],
    "lab_summary": "Parsed 8 lab results with 5 abnormal finding(s) including leukocytosis, low hemoglobin, hyponatremia, elevated creatinine, elevated troponin i.",
    "source_metadata": {
      "digital_table": true
    }
  },
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

Each `LabInput` also has a `source_type` and `content`:

```python
LabInput(source_type="digital_table", content="WBC 12.4 K/uL 4.0-10.5 H")
LabInput(source_type="image_report",  content="report.png")  # HF OCR when configured
LabInput(source_type="pdf_report",    content="report.pdf")  # pypdf for text PDFs
```

The Lab Agent uses report-provided reference ranges first and classifies abnormalities deterministically.

---

## Extending

- **Add a new entity category**: Update `ENTITY_LABEL_MAP` in `config.py` and add the field to `ClinicalEntities`.
- **Swap the ASR model**: Change `WHISPER_MODEL` in `config.py`.
- **Swap the summarization model**: Change `MEDICAL_T5_MODEL` in `config.py`.
- **Swap the lab OCR model**: Change `LAB_OCR_PRIMARY_MODEL` in `config.py`.
- **Expand lab terminology**: Update the synonym maps in `pipeline/lab_normalizer.py`.
