"""
Entity fusion & normalisation — merges ClinicalEntities from all input
sources into a single unified structure.

Post-fusion passes (applied after cross-source merge):
  1. Cross-source subsumption removal   ("pain" dropped when "chest pain" exists)
  2. Duration consolidation             fragments → canonical phrases
  3. Noise removal                      bare connectors stripped from durations
  4. Canonical display formatting       abbreviations uppercased consistently
  5. Final deduplication                catches any remaining duplicates
"""

import logging
import re

from models.clinical_bert_model import ABBREVIATION_MAP, SUBSUMPTION_MAP
from schemas.clinical_entities import ClinicalEntities

logger = logging.getLogger(__name__)

# Word-to-number for written-out numbers inside duration phrases
_WORD_TO_NUM: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}

# removed from the duration list when more specific phrases exist.
_DURATION_NOISE: set[str] = {
    "for the past", "since", "onset", "days", "weeks",
    "months", "years", "hours", "chronic", "acute",
}

# Fragments that should be lifted into severity, not left in durations
_DURATION_TO_SEVERITY: set[str] = {"acute", "chronic"}



def fuse_entities(entity_lists: list[ClinicalEntities]) -> ClinicalEntities:
    """
    Merge all ClinicalEntities in *entity_lists* into one unified instance
    and apply all post-fusion normalisation passes.

    Args:
        entity_lists: One ClinicalEntities per input source.

    Returns:
        A fully normalised, deduplicated :class:`ClinicalEntities`.
    """
    if not entity_lists:
        logger.warning("fuse_entities received empty list; returning empty entities.")
        return ClinicalEntities()

    # Guard: filter out None entries
    valid = [e for e in entity_lists if e is not None]
    if not valid:
        return ClinicalEntities()

    # Merge all sources
    unified = valid[0]
    for entities in valid[1:]:
        unified = unified.merge(entities)

    # Post-fusion normalisation pipeline
    unified = _normalise_abbreviations(unified)
    unified = _remove_subsumed_cross_source(unified)
    unified = _consolidate_durations(unified, unified.severity)
    unified = _remove_duration_noise(unified)
    unified = _ensure_no_nulls(unified)

    _log_summary(unified)
    return unified


# ── normalisation passes ──────────────────────────────────────────────────────

def _normalise_abbreviations(entities: ClinicalEntities) -> ClinicalEntities:
    """Apply ABBREVIATION_MAP to every entity in every category."""
    def norm_list(items: list[str]) -> list[str]:
        return [ABBREVIATION_MAP.get(i.lower(), i) for i in (items or [])]

    return ClinicalEntities(
        symptoms=_dedup(norm_list(entities.symptoms)),
        conditions=_dedup(norm_list(entities.conditions)),
        medications=_dedup(norm_list(entities.medications)),
        procedures=_dedup(norm_list(entities.procedures)),
        durations=_dedup(norm_list(entities.durations)),
        severity=_dedup(norm_list(entities.severity)),
    )


def _remove_subsumed_cross_source(entities: ClinicalEntities) -> ClinicalEntities:
    """
    Remove broad terms that are subsumed by more-specific terms in the
    same category — applied after cross-source merge.

    Example: ["chest pain", "pain"] → ["chest pain"]
    Example: ["type 2 diabetes", "diabetes"] → ["type 2 diabetes"]
    """
    return ClinicalEntities(
        symptoms=_drop_subsumed(entities.symptoms or []),
        conditions=_drop_subsumed(entities.conditions or []),
        medications=_drop_subsumed(entities.medications or []),
        procedures=_drop_subsumed(entities.procedures or []),
        durations=_drop_subsumed(entities.durations or []),
        severity=_drop_subsumed(entities.severity or []),
    )


def _drop_subsumed(items: list[str]) -> list[str]:
    """Remove items whose SUBSUMPTION_MAP specific-phrases are all present."""
    lower_set = {i.lower() for i in items}
    result: list[str] = []
    for item in items:
        specifics = SUBSUMPTION_MAP.get(item.lower(), set())
        if not specifics.intersection(lower_set):
            result.append(item)
    return result


def _consolidate_durations(
    entities: ClinicalEntities,
    severity_list: list[str],
) -> ClinicalEntities:
    """
    Consolidate raw duration tokens into canonical clinical phrases.

    Rules applied:
    • Written-out numbers → digits  ("three days" → "3 days")
    • Digit-hyphen-unit  → spaced   ("3-day" → "3 days")
    • Remove tokens that belong to severity (e.g. "acute", "chronic")
      from durations — they are already captured in severity.
    • Deduplicate consolidated phrases.
    """
    severity_lower = {s.lower() for s in (severity_list or [])}
    raw = list(entities.durations or [])
    consolidated: list[str] = []

    for phrase in raw:
        # Move severity-overlapping duration tokens to severity (they stay
        # in severity; we just drop them from durations here)
        if phrase.lower() in _DURATION_TO_SEVERITY and phrase.lower() in severity_lower:
            continue

        canonical = _canonicalise_duration(phrase)
        if canonical:
            consolidated.append(canonical)

    return ClinicalEntities(
        symptoms=entities.symptoms or [],
        conditions=entities.conditions or [],
        medications=entities.medications or [],
        procedures=entities.procedures or [],
        durations=_dedup(consolidated),
        severity=entities.severity or [],
    )


def _canonicalise_duration(phrase: str) -> str:
    """
    Convert a raw duration phrase to a canonical form.
    Returns empty string if the phrase is a bare noise fragment.
    """
    p = phrase.strip().lower()

    # Replace written-out numbers with digits
    for word, digit in _WORD_TO_NUM.items():
        p = re.sub(rf"\b{word}\b", digit, p)

    # "3-day" / "3-week" / "3-month" → "3 days" / "3 weeks" / "3 months"
    p = re.sub(r"(\d+)-day", r"\1 days", p)
    p = re.sub(r"(\d+)-week", r"\1 weeks", p)
    p = re.sub(r"(\d+)-month", r"\1 months", p)

    # "for the past N days/weeks" → "N days/weeks"
    p = re.sub(r"for the past\s+", "", p).strip()

    # "prn" already normalised to "as needed" by ABBREVIATION_MAP upstream

    # Strip bare noise connectors that carry no standalone meaning
    if p in _DURATION_NOISE:
        return ""

    # Title-case for display consistency
    return p.strip()


def _remove_duration_noise(entities: ClinicalEntities) -> ClinicalEntities:
    """Final pass: remove empty strings and bare noise from durations."""
    clean = [d for d in (entities.durations or []) if d and d not in _DURATION_NOISE]
    return ClinicalEntities(
        symptoms=entities.symptoms or [],
        conditions=entities.conditions or [],
        medications=entities.medications or [],
        procedures=entities.procedures or [],
        durations=_dedup(clean),
        severity=entities.severity or [],
    )


def _ensure_no_nulls(entities: ClinicalEntities) -> ClinicalEntities:
    """Guarantee every field is a list, never None."""
    return ClinicalEntities(
        symptoms=list(entities.symptoms or []),
        conditions=list(entities.conditions or []),
        medications=list(entities.medications or []),
        procedures=list(entities.procedures or []),
        durations=list(entities.durations or []),
        severity=list(entities.severity or []),
    )


# ── shared helpers ────────────────────────────────────────────────────────────

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


# ── logging ───────────────────────────────────────────────────────────────────

def _log_summary(entities: ClinicalEntities) -> None:
    total = sum(len(getattr(entities, f) or []) for f in
                ("symptoms", "conditions", "medications",
                 "procedures", "durations", "severity"))
    logger.info(
        "Entity fusion complete. Total unique entities: %d "
        "(symptoms=%d, conditions=%d, medications=%d, "
        "procedures=%d, durations=%d, severity=%d)",
        total,
        len(entities.symptoms or []),
        len(entities.conditions or []),
        len(entities.medications or []),
        len(entities.procedures or []),
        len(entities.durations or []),
        len(entities.severity or []),
    )