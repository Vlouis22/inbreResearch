"""
Entity fusion & normalization.

Merges ClinicalEntities extracted from multiple input sources into a
single unified ClinicalEntities instance, deduplicating across sources.
"""

import logging

from schemas.clinical_entities import ClinicalEntities

logger = logging.getLogger(__name__)


def fuse_entities(entity_lists: list[ClinicalEntities]) -> ClinicalEntities:
    """
    Merge all ClinicalEntities in *entity_lists* into one unified instance.

    Deduplication is case-insensitive and performed inside ClinicalEntities.merge().

    Args:
        entity_lists: One ClinicalEntities per input source.

    Returns:
        A single merged and deduplicated :class:`ClinicalEntities`.
    """
    if not entity_lists:
        logger.warning("fuse_entities received an empty list; returning empty entities.")
        return ClinicalEntities()

    unified = entity_lists[0]
    for entities in entity_lists[1:]:
        unified = unified.merge(entities)

    _log_summary(unified)
    return unified


def _log_summary(entities: ClinicalEntities) -> None:
    total = sum(
        len(v)
        for v in (
            entities.symptoms,
            entities.conditions,
            entities.medications,
            entities.procedures,
            entities.durations,
            entities.severity,
        )
    )
    logger.info(
        "Entity fusion complete. Total unique entities: %d "
        "(symptoms=%d, conditions=%d, medications=%d, procedures=%d, "
        "durations=%d, severity=%d)",
        total,
        len(entities.symptoms),
        len(entities.conditions),
        len(entities.medications),
        len(entities.procedures),
        len(entities.durations),
        len(entities.severity),
    )
