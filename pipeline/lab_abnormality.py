"""
Deterministic abnormality classification and lab summarization.
"""

from __future__ import annotations

import logging

from config import LAB_SUMMARY_MAX_FINDINGS
from schemas.lab_profile import LabResultRow

logger = logging.getLogger(__name__)

_SPECIAL_FINDINGS: dict[tuple[str, str], str] = {
    ("White Blood Cell Count", "high"): "leukocytosis",
    ("White Blood Cell Count", "low"): "leukopenia",
    ("Platelet Count", "low"): "thrombocytopenia",
    ("Platelet Count", "high"): "thrombocytosis",
    ("Sodium", "low"): "hyponatremia",
    ("Sodium", "high"): "hypernatremia",
    ("Potassium", "low"): "hypokalemia",
    ("Potassium", "high"): "hyperkalemia",
    ("Glucose", "high"): "hyperglycemia",
    ("Creatinine", "high"): "elevated creatinine",
    ("eGFR", "low"): "reduced eGFR",
    ("Troponin", "high"): "elevated troponin",
    ("Troponin I", "high"): "elevated troponin I",
    ("Troponin T", "high"): "elevated troponin T",
}


def classify_lab_rows(rows: list[LabResultRow]) -> tuple[list[LabResultRow], list[str], str]:
    """Classify each row and return enriched rows, deduplicated findings, and a summary."""
    enriched: list[LabResultRow] = []
    findings: list[str] = []

    for row in rows:
        flag = _compute_flag(row)
        enriched_row = LabResultRow(**row.model_dump())
        enriched_row.computed_flag = flag
        enriched.append(enriched_row)

        finding = _derive_finding(enriched_row)
        if finding:
            findings.append(finding)

    findings = _dedup(findings)
    summary = build_lab_summary(enriched, findings)
    logger.info("Computed abnormality classifications for %d lab row(s).", len(enriched))
    return enriched, findings, summary


def _compute_flag(row: LabResultRow) -> str:
    if row.reported_flag_optional == "critical":
        return "critical"

    if row.value_numeric_optional is not None:
        return _compute_numeric_flag(row)

    return _compute_qualitative_flag(row)


def _compute_numeric_flag(row: LabResultRow) -> str:
    value = row.value_numeric_optional
    assert value is not None

    if row.reference_low_optional is not None and row.reference_high_optional is not None:
        if value < row.reference_low_optional:
            return "low"
        if value > row.reference_high_optional:
            return "high"
        return "normal"

    ref = (row.reference_range_raw or "").strip()
    if not ref:
        return "unknown"

    if ref.startswith("<") and row.reference_high_optional is not None:
        return "normal" if value <= row.reference_high_optional else "high"
    if ref.startswith(">") and row.reference_low_optional is not None:
        return "normal" if value >= row.reference_low_optional else "low"

    return "unknown"


def _compute_qualitative_flag(row: LabResultRow) -> str:
    value = (row.value_raw or "").strip().lower()
    reference = (row.reference_range_raw or "").strip().lower()
    if not reference:
        return "unknown"
    if value == reference:
        return "normal"
    if reference == "negative" and value in {"positive", "trace", "small", "moderate", "large"}:
        return "abnormal"
    if reference == "normal" and value == "abnormal":
        return "abnormal"
    return "unknown"


def _derive_finding(row: LabResultRow) -> str | None:
    if row.computed_flag not in {"low", "high", "critical", "abnormal"}:
        return None

    canonical = row.test_name_canonical or row.test_name_raw
    key = (canonical, "high" if row.computed_flag == "critical" else row.computed_flag)
    if key in _SPECIAL_FINDINGS:
        return _SPECIAL_FINDINGS[key]

    if row.computed_flag == "high":
        return f"high {canonical.lower()}"
    if row.computed_flag == "low":
        return f"low {canonical.lower()}"
    if row.computed_flag == "critical":
        return f"critical {canonical.lower()}"
    return f"abnormal {canonical.lower()}"


def build_lab_summary(rows: list[LabResultRow], findings: list[str]) -> str:
    if not rows:
        return "No lab results were available."

    abnormal_count = sum(row.computed_flag in {"low", "high", "critical", "abnormal"} for row in rows)
    unknown_count = sum(row.computed_flag == "unknown" for row in rows)
    parts = [f"Parsed {len(rows)} lab results"]

    if abnormal_count:
        top_findings = ", ".join(findings[:LAB_SUMMARY_MAX_FINDINGS])
        parts.append(f"with {abnormal_count} abnormal finding(s)")
        if top_findings:
            parts.append(f"including {top_findings}")
    else:
        parts.append("with no computed abnormalities")

    if unknown_count:
        parts.append(f"and {unknown_count} result(s) left as unknown due to missing or non-numeric reference data")

    return " ".join(parts) + "."


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result
