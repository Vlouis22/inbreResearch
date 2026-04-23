"""
Normalization helpers for LabResultRow objects.
"""

from __future__ import annotations

import logging
import re

from schemas.lab_profile import LabResultRow

logger = logging.getLogger(__name__)

_TEST_SYNONYMS: dict[str, str] = {
    "wbc": "White Blood Cell Count",
    "white blood cell count": "White Blood Cell Count",
    "rbc": "Red Blood Cell Count",
    "red blood cell count": "Red Blood Cell Count",
    "hgb": "Hemoglobin",
    "hb": "Hemoglobin",
    "hemoglobin": "Hemoglobin",
    "hct": "Hematocrit",
    "hematocrit": "Hematocrit",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",
    "plt": "Platelet Count",
    "platelets": "Platelet Count",
    "platelet count": "Platelet Count",
    "neutrophils": "Neutrophils",
    "lymphocytes": "Lymphocytes",
    "na": "Sodium",
    "sodium": "Sodium",
    "k": "Potassium",
    "potassium": "Potassium",
    "chloride": "Chloride",
    "cl": "Chloride",
    "co2": "CO2",
    "bicarbonate": "CO2",
    "bun": "BUN",
    "blood urea nitrogen": "BUN",
    "creatinine": "Creatinine",
    "cr": "Creatinine",
    "egfr": "eGFR",
    "glucose": "Glucose",
    "calcium": "Calcium",
    "albumin": "Albumin",
    "total protein": "Total Protein",
    "bilirubin total": "Total Bilirubin",
    "total bilirubin": "Total Bilirubin",
    "bilirubin direct": "Direct Bilirubin",
    "direct bilirubin": "Direct Bilirubin",
    "alk phos": "Alkaline Phosphatase",
    "alp": "Alkaline Phosphatase",
    "alkaline phosphatase": "Alkaline Phosphatase",
    "ast": "AST",
    "alt": "ALT",
    "cholesterol": "Total Cholesterol",
    "total cholesterol": "Total Cholesterol",
    "hdl": "HDL Cholesterol",
    "ldl": "LDL Cholesterol",
    "triglycerides": "Triglycerides",
    "pt": "PT",
    "inr": "INR",
    "ptt": "PTT",
    "a1c": "HbA1c",
    "hba1c": "HbA1c",
    "troponin": "Troponin",
    "troponin i": "Troponin I",
    "troponin t": "Troponin T",
    "leukocyte esterase": "Urine Leukocyte Esterase",
    "nitrite": "Urine Nitrite",
    "protein": "Urine Protein",
    "glucose urine": "Urine Glucose",
    "ketones": "Urine Ketones",
    "blood": "Urine Blood",
}

_UNIT_SYNONYMS: dict[str, str] = {
    "k/ul": "10^3/uL",
    "k/µl": "10^3/uL",
    "10^3/ul": "10^3/uL",
    "x10^3/ul": "10^3/uL",
    "m/ul": "10^6/uL",
    "10^6/ul": "10^6/uL",
    "g/dl": "g/dL",
    "mg/dl": "mg/dL",
    "ng/ml": "ng/mL",
    "pg/ml": "pg/mL",
    "mmol/l": "mmol/L",
    "meq/l": "mEq/L",
    "u/l": "U/L",
    "iu/l": "IU/L",
    "sec": "sec",
    "secs": "sec",
    "seconds": "sec",
    "fl": "fL",
    "pg": "pg",
    "%": "%",
}

_PANEL_BY_TEST: dict[str, str] = {
    "White Blood Cell Count": "CBC",
    "Red Blood Cell Count": "CBC",
    "Hemoglobin": "CBC",
    "Hematocrit": "CBC",
    "MCV": "CBC",
    "MCH": "CBC",
    "MCHC": "CBC",
    "RDW": "CBC",
    "Platelet Count": "CBC",
    "Neutrophils": "CBC",
    "Lymphocytes": "CBC",
    "Sodium": "BMP",
    "Potassium": "BMP",
    "Chloride": "BMP",
    "CO2": "BMP",
    "BUN": "BMP",
    "Creatinine": "BMP",
    "eGFR": "BMP",
    "Glucose": "BMP",
    "Calcium": "BMP",
    "Albumin": "CMP",
    "Total Protein": "CMP",
    "Total Bilirubin": "LFT",
    "Direct Bilirubin": "LFT",
    "Alkaline Phosphatase": "LFT",
    "AST": "LFT",
    "ALT": "LFT",
    "Total Cholesterol": "lipid panel",
    "HDL Cholesterol": "lipid panel",
    "LDL Cholesterol": "lipid panel",
    "Triglycerides": "lipid panel",
    "PT": "coagulation",
    "INR": "coagulation",
    "PTT": "coagulation",
    "HbA1c": "HbA1c",
    "Troponin": "troponin",
    "Troponin I": "troponin",
    "Troponin T": "troponin",
    "Urine Leukocyte Esterase": "urinalysis",
    "Urine Nitrite": "urinalysis",
    "Urine Protein": "urinalysis",
    "Urine Glucose": "urinalysis",
    "Urine Ketones": "urinalysis",
    "Urine Blood": "urinalysis",
}

_QUALITATIVE_VALUES = {
    "positive", "negative", "trace", "small", "moderate", "large",
    "normal", "abnormal", "clear", "cloudy",
}


def normalize_lab_rows(rows: list[LabResultRow]) -> list[LabResultRow]:
    normalized: list[LabResultRow] = []
    for row in rows:
        canonical = _canonicalize_test_name(row.test_name_raw)
        unit = _canonicalize_unit(row.unit_raw)
        low, high = _parse_reference_bounds(row.reference_range_raw)
        numeric_value = _parse_numeric_value(row.value_raw)
        reported_flag = _normalize_reported_flag(row.reported_flag_optional)
        panel_name = _normalize_panel(row.panel_name, canonical)
        confidence = _adjust_confidence(row.confidence, canonical, numeric_value, row.reference_range_raw)

        normalized.append(
            LabResultRow(
                panel_name=panel_name,
                test_name_raw=row.test_name_raw.strip(),
                test_name_canonical=canonical,
                test_code_optional=None,
                value_raw=row.value_raw.strip(),
                value_numeric_optional=numeric_value,
                comparator_optional=row.comparator_optional,
                unit_raw=row.unit_raw.strip(),
                unit_canonical=unit,
                reference_range_raw=row.reference_range_raw.strip(),
                reference_low_optional=low,
                reference_high_optional=high,
                reported_flag_optional=reported_flag,
                computed_flag=row.computed_flag,
                specimen_optional=_normalize_specimen(row.specimen_optional),
                confidence=confidence,
                source_page_or_row=row.source_page_or_row,
            )
        )

    logger.info("Normalized %d lab row(s).", len(normalized))
    return normalized


def _canonicalize_test_name(raw: str) -> str:
    key = _normalize_key(raw)
    return _TEST_SYNONYMS.get(key, raw.strip())


def _canonicalize_unit(raw: str) -> str:
    key = _normalize_key(raw)
    return _UNIT_SYNONYMS.get(key, raw.strip())


def _normalize_key(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_numeric_value(value_raw: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", value_raw or "")
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_reference_bounds(reference: str) -> tuple[float | None, float | None]:
    ref = (reference or "").strip().lower()
    if not ref:
        return None, None

    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)", ref)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    upper_match = re.search(r"<=?\s*(-?\d+(?:\.\d+)?)", ref)
    if upper_match:
        return None, float(upper_match.group(1))

    lower_match = re.search(r">=?\s*(-?\d+(?:\.\d+)?)", ref)
    if lower_match:
        return float(lower_match.group(1)), None

    return None, None


def _normalize_reported_flag(flag: str | None) -> str | None:
    if not flag:
        return None
    key = flag.strip().upper()
    if key in {"H", "HIGH"}:
        return "high"
    if key in {"L", "LOW"}:
        return "low"
    if key in {"HH", "LL", "CRITICAL", "PANIC"}:
        return "critical"
    if key in {"N", "NORMAL"}:
        return "normal"
    if key in {"A", "ABNORMAL", "POSITIVE"}:
        return "abnormal"
    if key == "NEGATIVE":
        return "normal"
    return flag.strip().lower()


def _normalize_panel(panel_name: str, canonical_test: str) -> str:
    panel = panel_name.strip()
    if panel:
        lowered = panel.lower()
        if "complete blood count" in lowered:
            return "CBC"
        if "basic metabolic panel" in lowered:
            return "BMP"
        if "comprehensive metabolic panel" in lowered:
            return "CMP"
        if "liver" in lowered:
            return "LFT"
        if "lipid" in lowered:
            return "lipid panel"
        if "coag" in lowered:
            return "coagulation"
        if "urinalysis" in lowered:
            return "urinalysis"
        return panel
    return _PANEL_BY_TEST.get(canonical_test, "")


def _normalize_specimen(specimen: str | None) -> str | None:
    if not specimen:
        return None
    normalized = specimen.strip().lower()
    return normalized if normalized else None


def _adjust_confidence(
    base: float,
    canonical_test: str,
    numeric_value: float | None,
    reference_range_raw: str,
) -> float:
    confidence = max(0.0, min(base or 0.0, 1.0))
    if canonical_test and canonical_test != "":
        confidence += 0.03
    if numeric_value is not None:
        confidence += 0.03
    elif (reference_range_raw or "").strip().lower() in _QUALITATIVE_VALUES:
        confidence += 0.02
    return round(min(confidence, 0.99), 2)
