"""
Entity extractor — Pipeline 1.

Uses ClinicalBERT's run_ner() output to populate a ClinicalEntities
instance for a single normalized text source.
"""

import logging

from config import ENTITY_LABEL_MAP, NER_CONFIDENCE_THRESHOLD
from models.clinical_bert_model import run_ner
from schemas.clinical_entities import ClinicalEntities

logger = logging.getLogger(__name__)


def extract_entities(text: str) -> ClinicalEntities:
    """
    Run entity extraction on *text* and map labels to ClinicalEntities.

    Args:
        text: Normalized clinical text from a single source.

    Returns:
        A :class:`ClinicalEntities` populated with entities found in *text*.
    """
    if not text.strip():
        logger.warning("extract_entities received empty text; returning empty entities.")
        return ClinicalEntities()

    raw_entities: list[dict] = run_ner(text)

    buckets: dict[str, list[str]] = {
        "symptoms": [],
        "conditions": [],
        "medications": [],
        "procedures": [],
        "durations": [],
        "severity": [],
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
            buckets[category].append(word)

    result = ClinicalEntities(**buckets)
    logger.debug(
        "Extracted entities — symptoms:%d conditions:%d medications:%d "
        "procedures:%d durations:%d severity:%d",
        len(result.symptoms), len(result.conditions), len(result.medications),
        len(result.procedures), len(result.durations), len(result.severity),
    )
    return result


def _resolve_category(label: str) -> str | None:
    """
    Map a raw NER label to one of the six canonical ClinicalEntities categories.
    Tries exact match, then BIO-prefix-stripped match, then partial match.
    """
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