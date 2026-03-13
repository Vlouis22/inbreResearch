"""
Clinical Text Agent — main entry point.

Demonstrates the full pipeline end-to-end:

  1.  Receive clinical inputs (text or audio paths)
  2.  Convert audio inputs to text via ASR
  3.  Normalize all text
  4.  Run ClinicalBERT entity extraction (per source)          [Pipeline 1]
  5.  Fuse entities from all sources
  6.  Combine texts and generate Medical T5 summary            [Pipeline 2]
  7.  Merge into a Structured Patient Profile
  8.  Print the profile as formatted JSON

The pipeline stops at the Structured Patient Profile.
Diagnostic reasoning is NOT implemented here.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when running as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from config import CLINICAL_BERT_MODEL, MEDICAL_T5_MODEL, WHISPER_MODEL
from models.asr_model import load_asr_model, transcribe_audio
from models.clinical_bert_model import load_clinical_bert
from models.medical_t5_model import load_medical_t5
from pipeline.entity_extractor import extract_entities
from pipeline.entity_fusion import fuse_entities
from pipeline.summarizer import generate_summary
from pipeline.text_normalizer import normalize
from schemas.clinical_entities import ClinicalEntities
from schemas.structured_patient_profile import SourceMetadata, StructuredPatientProfile
from utils.text_utils import is_audio_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input data class
# ---------------------------------------------------------------------------

@dataclass
class ClinicalInput:
    """Represents a single clinical input source."""
    source_type: str   # "doctor_notes" | "patient_conversation" | "health_records"
    content: str       # raw text OR path to an audio file


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_all_models() -> None:
    """Load all models required by the pipeline."""
    logger.info("=== Loading models ===")
    load_asr_model(WHISPER_MODEL)
    load_clinical_bert(CLINICAL_BERT_MODEL)
    load_medical_t5(MEDICAL_T5_MODEL)
    logger.info("=== All models ready ===")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_asr(inputs: list[ClinicalInput]) -> list[tuple[ClinicalInput, str]]:
    """
    Stage 2 & 3 — Inspect each input and convert audio to text where needed.

    Returns a list of (input, raw_text) pairs.
    """
    results: list[tuple[ClinicalInput, str]] = []
    for inp in inputs:
        if is_audio_file(inp.content):
            logger.info("[ASR] Transcribing audio input (%s): %s", inp.source_type, inp.content)
            raw_text = transcribe_audio(inp.content)
        else:
            logger.info("[TEXT] Using text input (%s).", inp.source_type)
            raw_text = inp.content
        results.append((inp, raw_text))
    return results


def stage_normalize(pairs: list[tuple[ClinicalInput, str]]) -> list[tuple[ClinicalInput, str]]:
    """Stage 4 — Normalize all raw texts."""
    return [(inp, normalize(raw)) for inp, raw in pairs]


def stage_entity_extraction(
    normalized_pairs: list[tuple[ClinicalInput, str]]
) -> list[ClinicalEntities]:
    """Stage 5 — Run ClinicalBERT NER on each normalized text source."""
    entity_lists: list[ClinicalEntities] = []
    for inp, text in normalized_pairs:
        logger.info("[NER] Extracting entities from: %s", inp.source_type)
        entities = extract_entities(text)
        entity_lists.append(entities)
    return entity_lists


def stage_entity_fusion(entity_lists: list[ClinicalEntities]) -> ClinicalEntities:
    """Stage 6 — Fuse all per-source entity lists into unified entities."""
    logger.info("[FUSION] Merging entity lists from %d source(s).", len(entity_lists))
    return fuse_entities(entity_lists)


def stage_summarization(normalized_pairs: list[tuple[ClinicalInput, str]]) -> str:
    """Stages 7 & 8 — Combine all normalized texts and generate the medical summary."""
    texts = [text for _, text in normalized_pairs]
    logger.info("[SUMMARIZATION] Generating Medical T5 summary from %d source(s).", len(texts))
    return generate_summary(texts)


def stage_build_profile(
    unified_entities: ClinicalEntities,
    medical_summary: str,
    inputs: list[ClinicalInput],
) -> StructuredPatientProfile:
    """Stage 9 & 10 — Merge pipeline outputs into the Structured Patient Profile."""
    source_types = {inp.source_type for inp in inputs}
    metadata = SourceMetadata(
        doctor_notes="doctor_notes" in source_types,
        patient_conversation="patient_conversation" in source_types,
        health_records="health_records" in source_types,
    )
    return StructuredPatientProfile(
        clinical_entities=unified_entities,
        medical_summary=medical_summary,
        source_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(inputs: list[ClinicalInput]) -> StructuredPatientProfile:
    """
    Execute the full Clinical Text Agent pipeline on *inputs*.

    Args:
        inputs: List of ClinicalInput objects (text or audio).

    Returns:
        A :class:`StructuredPatientProfile` — the terminal output.
    """
    if not inputs:
        raise ValueError("At least one ClinicalInput is required.")

    # Stage 2 & 3: ASR
    raw_pairs = stage_asr(inputs)

    # Stage 4: Normalize
    normalized_pairs = stage_normalize(raw_pairs)

    # Stage 5: Entity extraction (Pipeline 1)
    entity_lists = stage_entity_extraction(normalized_pairs)

    # Stage 6: Entity fusion
    unified_entities = stage_entity_fusion(entity_lists)

    # Stages 7 & 8: Summarization (Pipeline 2)
    medical_summary = stage_summarization(normalized_pairs)

    # Stages 9 & 10: Assemble Structured Patient Profile
    profile = stage_build_profile(unified_entities, medical_summary, inputs)

    return profile


# ---------------------------------------------------------------------------
# Example inputs
# ---------------------------------------------------------------------------

EXAMPLE_INPUTS: list[ClinicalInput] = [
    # ClinicalInput(
    #     source_type="doctor_notes",
    #     content=(
    #         "Patient is a 58-year-old male presenting with a 3-day history of severe chest pain "
    #         "and shortness of breath. History of hypertension and type 2 diabetes. "
    #         "Current medications include metformin 1000 mg twice daily and lisinopril 10 mg daily. "
    #         "Ordered ECG and chest X-ray. Elevated troponin levels noted."
    #     ),
    # ),
    ClinicalInput(
        #source_type="patient_conversation",
        # Simulated audio path — ASR placeholder will activate
        #content="patient_voice_note.wav",
        source_type="doctor_notes",
        content="clinical_text_agent/doctor_note.wav"
    ),
    ClinicalInput(
        source_type="health_records",
        content=(
            "Previous hospital admission 6 months ago for moderate pneumonia. "
            "Spirometry showed mild COPD. Prescribed albuterol inhaler PRN. "
            "No known drug allergies. Blood pressure consistently above 140/90 mmHg. "
            "Latest HbA1c: 7.8%. MRI brain — no acute findings."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Clinical Text Agent — Structured Patient Profile Generator")
    print("=" * 70)

    # Load all models (with graceful fallback if unavailable)
    load_all_models()

    logger.info("Processing %d clinical input(s)...", len(EXAMPLE_INPUTS))
    profile: StructuredPatientProfile = run_pipeline(EXAMPLE_INPUTS)

    # Output the Structured Patient Profile as formatted JSON
    print("\n" + "=" * 70)
    print("  STRUCTURED PATIENT PROFILE")
    print("=" * 70)
    print(profile.model_dump_json(indent=2))
    print("=" * 70)
    print("  Pipeline complete. Diagnostic reasoning is handled by a separate system.")
    print("=" * 70)


if __name__ == "__main__":
    main()
