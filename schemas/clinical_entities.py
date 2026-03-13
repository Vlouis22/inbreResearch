"""
Pydantic schema for extracted clinical entities.
"""

from pydantic import BaseModel, Field


class ClinicalEntities(BaseModel):
    """Structured container for all entity categories extracted from clinical text."""

    symptoms: list[str] = Field(default_factory=list, description="Reported symptoms")
    conditions: list[str] = Field(default_factory=list, description="Diagnoses and medical conditions")
    medications: list[str] = Field(default_factory=list, description="Current and past medications")
    procedures: list[str] = Field(default_factory=list, description="Clinical procedures and tests")
    durations: list[str] = Field(default_factory=list, description="Time references and durations")
    severity: list[str] = Field(default_factory=list, description="Severity or degree indicators")

    def merge(self, other: "ClinicalEntities") -> "ClinicalEntities":
        """Return a new ClinicalEntities that combines self and other, deduplicating each field."""
        return ClinicalEntities(
            symptoms=_dedup(self.symptoms + other.symptoms),
            conditions=_dedup(self.conditions + other.conditions),
            medications=_dedup(self.medications + other.medications),
            procedures=_dedup(self.procedures + other.procedures),
            durations=_dedup(self.durations + other.durations),
            severity=_dedup(self.severity + other.severity),
        )


def _dedup(items: list[str]) -> list[str]:
    """Case-insensitive deduplication preserving insertion order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result
