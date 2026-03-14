"""
Entity extractor — Pipeline 1.

Runs ClinicalBERT NER on a single normalized text source, maps raw
labels to ClinicalEntities categories, then applies per-source
post-processing:

  • Abbreviation normalisation  (ecg → ECG, hba1c → HbA1c, …)
  • Subsumption removal          (removes "pain" when "chest pain" present)
  • Intra-category deduplication
"""

import logging

from config import ENTITY_LABEL_MAP, NER_CONFIDENCE_THRESHOLD
from models.clinical_bert_model import ABBREVIATION_MAP, SUBSUMPTION_MAP, run_ner
from schemas.clinical_entities import ClinicalEntities

logger = logging.getLogger(__name__)


def extract_entities(text: str) -> ClinicalEntities:
    """
    Extract and normalise clinical entities from a single *text* source.

    Args:
        text: Normalized clinical text from one input source.

    Returns:
        A :class:`ClinicalEntities` with cleaned, normalised entities.
    """
    if not text.strip():
        logger.warning("extract_entities received empty text; returning empty entities.")
        return ClinicalEntities()

    raw_entities: list[dict] = run_ner(text)

    buckets: dict[str, list[str]] = {
        "symptoms": [], "conditions": [], "medications": [],
        "procedures": [], "durations": [], "severity": [],
    }

    for entity in raw_entities:
        score: float = entity.get("score", 0.0)
        if score < NER_CONFIDENCE_THRESHOLD:
            continue

        label: str = entity.get("entity_group") or entity.get("entity", "")
        word: str = entity.get("word", "").strip()

        if not word or not label:
            continue

        category = _resolve_category(label)
        if category and category in buckets:
            # Normalise abbreviation before storing
            normalised = _normalise_term(word)
            buckets[category].append(normalised)

    # Per-source post-processing
    for cat in buckets:
        buckets[cat] = _dedup(buckets[cat])
        buckets[cat] = _remove_subsumed(buckets[cat])

    result = ClinicalEntities(**buckets)
    logger.debug(
        "Extracted — symptoms:%d conditions:%d medications:%d "
        "procedures:%d durations:%d severity:%d",
        len(result.symptoms), len(result.conditions), len(result.medications),
        len(result.procedures), len(result.durations), len(result.severity),
    )
    return result


def _resolve_category(label: str) -> str | None:
    """Map a raw NER label → canonical ClinicalEntities category name."""
    if label in ENTITY_LABEL_MAP:
        return ENTITY_LABEL_MAP[label]
    stripped = label[2:] if len(label) > 2 and label[1] == "-" else label
    if stripped in ENTITY_LABEL_MAP:
        return ENTITY_LABEL_MAP[stripped]
    upper = stripped.upper()
    for key, category in ENTITY_LABEL_MAP.items():
        if key in upper or upper in key:
            return category
    return None


def _normalise_term(term: str) -> str:
    """
    Apply abbreviation normalisation.
    Lookup is case-insensitive; the canonical form from ABBREVIATION_MAP is used.
    """
    return ABBREVIATION_MAP.get(term.lower(), term)


def _dedup(items: list[str]) -> list[str]:
    """Case-insensitive deduplication preserving insertion order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _remove_subsumed(items: list[str]) -> list[str]:
    """
    Remove a broad term if a more-specific term that subsumes it is present.

    Example: if ["chest pain", "pain"] → drop "pain" because SUBSUMPTION_MAP
    says "pain" is subsumed by "chest pain".
    """
    lower_items = {i.lower() for i in items}
    result: list[str] = []
    for item in items:
        specific_terms = SUBSUMPTION_MAP.get(item.lower(), set())
        # Keep item only if none of its specific superseding terms are present
        if not specific_terms.intersection(lower_items):
            result.append(item)
    return result