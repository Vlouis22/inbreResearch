"""
Pydantic schema for extracted clinical entities.
All list fields default to empty list and are validated to never be None,
ensuring downstream code can always iterate without null-checks.
"""

from typing import Any

try:
    from pydantic import BaseModel, Field, model_validator
    _HAS_MODEL_VALIDATOR = True
except ImportError:
    from pydantic import BaseModel, Field  # type: ignore
    _HAS_MODEL_VALIDATOR = False


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


if _HAS_MODEL_VALIDATOR:
    class ClinicalEntities(BaseModel):
        """Structured container for all entity categories extracted from clinical text."""

        symptoms:    list[str] = Field(default_factory=list, description="Reported symptoms")
        conditions:  list[str] = Field(default_factory=list, description="Diagnoses and medical conditions")
        medications: list[str] = Field(default_factory=list, description="Current and past medications")
        procedures:  list[str] = Field(default_factory=list, description="Clinical procedures and tests")
        durations:   list[str] = Field(default_factory=list, description="Time references and durations")
        severity:    list[str] = Field(default_factory=list, description="Severity or degree indicators")

        @model_validator(mode="before")
        @classmethod
        def coerce_none_to_empty_list(cls, values: Any) -> Any:
            """Replace any None field value with an empty list before validation."""
            if isinstance(values, dict):
                for field in ("symptoms", "conditions", "medications",
                              "procedures", "durations", "severity"):
                    if values.get(field) is None:
                        values[field] = []
            return values

        def merge(self, other: "ClinicalEntities") -> "ClinicalEntities":
            return ClinicalEntities(
                symptoms=_dedup((self.symptoms or []) + (other.symptoms or [])),
                conditions=_dedup((self.conditions or []) + (other.conditions or [])),
                medications=_dedup((self.medications or []) + (other.medications or [])),
                procedures=_dedup((self.procedures or []) + (other.procedures or [])),
                durations=_dedup((self.durations or []) + (other.durations or [])),
                severity=_dedup((self.severity or []) + (other.severity or [])),
            )

else:
    # Fallback for older pydantic / mock environments — null-coercion handled in __init__
    class ClinicalEntities(BaseModel):  # type: ignore[no-redef]
        """Structured container for all entity categories extracted from clinical text."""

        symptoms:    list[str] = Field(default_factory=list, description="Reported symptoms")
        conditions:  list[str] = Field(default_factory=list, description="Diagnoses and medical conditions")
        medications: list[str] = Field(default_factory=list, description="Current and past medications")
        procedures:  list[str] = Field(default_factory=list, description="Clinical procedures and tests")
        durations:   list[str] = Field(default_factory=list, description="Time references and durations")
        severity:    list[str] = Field(default_factory=list, description="Severity or degree indicators")

        def __init__(self, **data: Any):
            # Coerce None → [] before the parent __init__ runs
            for field in ("symptoms", "conditions", "medications",
                          "procedures", "durations", "severity"):
                if data.get(field) is None:
                    data[field] = []
            super().__init__(**data)

        def merge(self, other: "ClinicalEntities") -> "ClinicalEntities":
            return ClinicalEntities(
                symptoms=_dedup((self.symptoms or []) + (other.symptoms or [])),
                conditions=_dedup((self.conditions or []) + (other.conditions or [])),
                medications=_dedup((self.medications or []) + (other.medications or [])),
                procedures=_dedup((self.procedures or []) + (other.procedures or [])),
                durations=_dedup((self.durations or []) + (other.durations or [])),
                severity=_dedup((self.severity or []) + (other.severity or [])),
            )