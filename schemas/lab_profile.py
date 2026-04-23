"""
Pydantic schemas for structured laboratory extraction.
"""

from typing import Any

try:
    from pydantic import BaseModel, Field, model_validator
    _HAS_MODEL_VALIDATOR = True
except ImportError:
    from pydantic import BaseModel, Field  # type: ignore
    _HAS_MODEL_VALIDATOR = False


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


_LIST_FIELDS = ("lab_results", "abnormal_findings")


class LabResultRow(BaseModel):
    """One extracted lab result row with raw and normalized fields."""

    panel_name: str = Field(default="", description="Panel heading when present")
    test_name_raw: str = Field(default="", description="Original test name surface form")
    test_name_canonical: str = Field(default="", description="Project-level canonical test name")
    test_code_optional: str | None = Field(default=None, description="Optional standard code")
    value_raw: str = Field(default="", description="Original value as seen in the report")
    value_numeric_optional: float | None = Field(default=None, description="Parsed numeric value when available")
    comparator_optional: str | None = Field(default=None, description="Optional comparator such as < or >")
    unit_raw: str = Field(default="", description="Original unit surface form")
    unit_canonical: str = Field(default="", description="Normalized unit string")
    reference_range_raw: str = Field(default="", description="Original reference range text")
    reference_low_optional: float | None = Field(default=None, description="Parsed lower bound when available")
    reference_high_optional: float | None = Field(default=None, description="Parsed upper bound when available")
    reported_flag_optional: str | None = Field(default=None, description="Flag reported by the source, e.g. H/L")
    computed_flag: str = Field(default="unknown", description="Deterministic abnormality classification")
    specimen_optional: str | None = Field(default=None, description="Optional specimen such as blood or urine")
    confidence: float = Field(default=0.0, description="Extraction confidence from 0.0 to 1.0")
    source_page_or_row: str = Field(default="", description="Traceability pointer to the source row or line")


class LabSourceMetadata(BaseModel):
    """Tracks the source formats used during the lab run."""

    pdf_report: bool = False
    image_report: bool = False
    digital_table: bool = False
    text_report: bool = False
    fallback_parser_used: bool = False
    ocr_provider: str = ""


if _HAS_MODEL_VALIDATOR:
    class LabProfile(BaseModel):
        """Final structured output of the Lab Agent."""

        lab_results: list[LabResultRow] = Field(default_factory=list)
        abnormal_findings: list[str] = Field(default_factory=list)
        lab_summary: str = Field(default="", description="Deterministic summary of lab abnormalities")
        source_metadata: LabSourceMetadata = Field(default_factory=LabSourceMetadata)

        @model_validator(mode="before")
        @classmethod
        def coerce_none_to_empty_list(cls, values: Any) -> Any:
            if isinstance(values, dict):
                for field in _LIST_FIELDS:
                    if values.get(field) is None:
                        values[field] = []
            return values

        @model_validator(mode="after")
        def deduplicate_findings(self) -> "LabProfile":
            self.abnormal_findings = _dedup(list(self.abnormal_findings or []))
            return self
else:
    class LabProfile(BaseModel):  # type: ignore[no-redef]
        """Fallback schema for environments without pydantic v2."""

        lab_results: list[LabResultRow] = Field(default_factory=list)
        abnormal_findings: list[str] = Field(default_factory=list)
        lab_summary: str = Field(default="", description="Deterministic summary of lab abnormalities")
        source_metadata: LabSourceMetadata = Field(default_factory=LabSourceMetadata)

        def __init__(self, **data: Any):
            for field in _LIST_FIELDS:
                if data.get(field) is None:
                    data[field] = []
            super().__init__(**data)
            self.abnormal_findings = _dedup(list(self.abnormal_findings or []))
