"""
Self-contained pipeline test.
Mocks transformers / torch / pydantic so the full pipeline logic can be
verified without any installed packages.
"""

import json
import sys
import re
import types
from pathlib import Path

# ── point imports at the project root ──────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── minimal pydantic v2 stub ────────────────────────────────────────────────
class _Field:
    def __init__(self, default=None, default_factory=None, description=""):
        self.default = default
        self.default_factory = default_factory

def Field(default=None, default_factory=None, description=""):
    return _Field(default=default, default_factory=default_factory, description=description)

class BaseModel:
    def __init__(self, **kwargs):
        for name, annotation in self.__class__.__annotations__.items():
            field_default = getattr(self.__class__, name, None)
            if isinstance(field_default, _Field):
                if name in kwargs:
                    setattr(self, name, kwargs[name])
                elif field_default.default_factory is not None:
                    setattr(self, name, field_default.default_factory())
                else:
                    setattr(self, name, field_default.default)
            else:
                setattr(self, name, kwargs.get(name, field_default))

    def model_dump(self):
        result = {}
        for name in self.__class__.__annotations__:
            val = getattr(self, name)
            if isinstance(val, BaseModel):
                result[name] = val.model_dump()
            elif isinstance(val, list):
                result[name] = val
            else:
                result[name] = val
        return result

    def model_dump_json(self, indent=2):
        return json.dumps(self.model_dump(), indent=indent)

pydantic_stub = types.ModuleType("pydantic")
pydantic_stub.BaseModel = BaseModel
pydantic_stub.Field = Field
sys.modules["pydantic"] = pydantic_stub


# ── transformers stub ───────────────────────────────────────────────────────
def _ner_pipeline_fn(text):
    """Simulate ClinicalBERT NER output using keyword rules."""
    lower = text.lower()
    hits = []
    keyword_map = [
        (["chest pain", "shortness of breath", "fever", "cough", "fatigue",
          "nausea", "dizziness", "headache", "weakness", "swelling"], "SYMPTOM"),
        (["hypertension", "diabetes", "pneumonia", "copd", "asthma",
          "infection", "heart failure", "anemia"], "DISEASE"),
        (["metformin", "lisinopril", "aspirin", "albuterol", "warfarin",
          "atorvastatin", "insulin", "ibuprofen", "amoxicillin"], "DRUG"),
        (["ecg", "x-ray", "mri", "ct scan", "spirometry", "blood test",
          "urinalysis", "biopsy", "ultrasound"], "PROCEDURE"),
        (["3-day", "6 months", "daily", "twice daily", "chronic", "acute",
          "weeks", "months", "years"], "DURATION"),
        (["severe", "mild", "moderate", "critical", "worsening",
          "stable", "elevated"], "SEVERITY"),
    ]
    seen = set()
    for keywords, label in keyword_map:
        for kw in keywords:
            if kw in lower and kw not in seen:
                seen.add(kw)
                hits.append({"word": kw, "entity_group": label, "score": 0.92,
                             "start": lower.index(kw), "end": lower.index(kw) + len(kw)})
    return hits

def _summarization_pipeline_fn(text, max_length=256, min_length=64, do_sample=False):
    """Simulate Medical T5 extractive summarization."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:4])
    if len(summary) > max_length * 5:
        summary = summary[: max_length * 5]
    return [{"summary_text": summary}]

class _MockPipeline:
    def __init__(self, task, **kwargs):
        self.task = task

    def __call__(self, text, **kwargs):
        if self.task == "ner":
            return _ner_pipeline_fn(text)
        if self.task == "summarization":
            return _summarization_pipeline_fn(text, **kwargs)
        if self.task == "automatic-speech-recognition":
            return {"text": (
                "Patient reports shortness of breath and chest pain for three days. "
                "No fever. Currently taking aspirin 81 mg daily."
            )}
        return {}

class _MockTokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

transformers_stub = types.ModuleType("transformers")
transformers_stub.pipeline = lambda task, **kw: _MockPipeline(task, **kw)
transformers_stub.AutoTokenizer = _MockTokenizer
sys.modules["transformers"] = transformers_stub


# ── now import the real project modules ────────────────────────────────────
from config import CLINICAL_BERT_MODEL, MEDICAL_T5_MODEL, WHISPER_MODEL  # noqa: E402
from models.asr_model import load_asr_model, transcribe_audio             # noqa: E402
from models.clinical_bert_model import load_clinical_bert, run_ner        # noqa: E402
from models.medical_t5_model import load_medical_t5, summarize            # noqa: E402
from pipeline.text_normalizer import normalize                             # noqa: E402
from pipeline.entity_extractor import extract_entities                     # noqa: E402
from pipeline.entity_fusion import fuse_entities                           # noqa: E402
from pipeline.summarizer import generate_summary                           # noqa: E402
from schemas.clinical_entities import ClinicalEntities                     # noqa: E402
from schemas.structured_patient_profile import SourceMetadata, StructuredPatientProfile  # noqa: E402
from utils.text_utils import is_audio_file                                 # noqa: E402
from main import ClinicalInput, run_pipeline, EXAMPLE_INPUTS              # noqa: E402


# ── run ─────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  Clinical Text Agent — Pipeline Test (mocked external dependencies)")
print("=" * 70)

# Load all models (uses mock transformers)
load_asr_model(WHISPER_MODEL)
load_clinical_bert(CLINICAL_BERT_MODEL)
load_medical_t5(MEDICAL_T5_MODEL)

# Execute the full pipeline
profile: StructuredPatientProfile = run_pipeline(EXAMPLE_INPUTS)

print("\n" + "=" * 70)
print("  STRUCTURED PATIENT PROFILE")
print("=" * 70)
output = profile.model_dump_json(indent=2)
print(output)
print("=" * 70)
print("  Pipeline COMPLETE — Diagnostic Reasoning Agent not implemented here.")
print("=" * 70)

# ── basic assertions ─────────────────────────────────────────────────────────
data = json.loads(output)
assert "clinical_entities" in data, "Missing clinical_entities"
assert "medical_summary" in data, "Missing medical_summary"
assert "source_metadata" in data, "Missing source_metadata"

entities = data["clinical_entities"]
assert all(k in entities for k in ["symptoms","conditions","medications",
                                    "procedures","durations","severity"]), \
    "Missing entity category"

metadata = data["source_metadata"]
assert metadata["doctor_notes"] is True,         "doctor_notes flag wrong"
assert metadata["patient_conversation"] is True, "patient_conversation flag wrong"
assert metadata["health_records"] is True,       "health_records flag wrong"

assert len(data["medical_summary"]) > 10, "Summary too short"
assert len(entities["symptoms"]) > 0,      "No symptoms extracted"
assert len(entities["conditions"]) > 0,    "No conditions extracted"
assert len(entities["medications"]) > 0,   "No medications extracted"

print("\n  ✓ All assertions passed.")
