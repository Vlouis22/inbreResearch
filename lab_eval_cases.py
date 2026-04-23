"""
Synthetic gold evaluation cases for the Lab Agent.

The dataset intentionally mixes clean tables, simulated OCR text, and harder
formats such as CSV, pipe tables, comparator-based ranges, and qualitative
urinalysis findings.
"""

from __future__ import annotations


def _expected(
    test: str,
    value: float | None,
    unit: str,
    flag: str,
) -> dict:
    return {
        "test_name_canonical": test,
        "value_numeric_optional": value,
        "unit_canonical": unit,
        "computed_flag": flag,
    }


def _make_case(
    report_id: str,
    source_type: str,
    content: str,
    expected_rows: list[dict],
    expected_findings: list[str],
) -> dict:
    return {
        "report_id": report_id,
        "source_type": source_type,
        "content": content,
        "expected_rows": expected_rows,
        "expected_findings": expected_findings,
    }


_CLEAN_CASES = [
    _make_case(
        "clean-01",
        "digital_table",
        (
            "CBC\n"
            "Test Result Unit Reference Range Flag\n"
            "WBC 12.4 K/uL 4.0-10.5 H\n"
            "Hemoglobin 10.2 g/dL 12.0-15.5 L\n"
            "Platelets 265 K/uL 150-400 N\n"
        ),
        [
            _expected("White Blood Cell Count", 12.4, "10^3/uL", "high"),
            _expected("Hemoglobin", 10.2, "g/dL", "low"),
            _expected("Platelet Count", 265.0, "10^3/uL", "normal"),
        ],
        ["leukocytosis", "low hemoglobin"],
    ),
    _make_case(
        "clean-02",
        "digital_table",
        (
            "BMP\n"
            "Sodium 132 mmol/L 135-145 L\n"
            "Potassium 4.7 mmol/L 3.5-5.1 N\n"
            "Creatinine 1.6 mg/dL 0.6-1.3 H\n"
        ),
        [
            _expected("Sodium", 132.0, "mmol/L", "low"),
            _expected("Potassium", 4.7, "mmol/L", "normal"),
            _expected("Creatinine", 1.6, "mg/dL", "high"),
        ],
        ["hyponatremia", "elevated creatinine"],
    ),
    _make_case(
        "clean-03",
        "digital_table",
        (
            "LFT\n"
            "AST 68 U/L 10-40 H\n"
            "ALT 74 U/L 7-56 H\n"
            "Albumin 3.1 g/dL 3.5-5.0 L\n"
        ),
        [
            _expected("AST", 68.0, "U/L", "high"),
            _expected("ALT", 74.0, "U/L", "high"),
            _expected("Albumin", 3.1, "g/dL", "low"),
        ],
        ["high ast", "high alt", "low albumin"],
    ),
    _make_case(
        "clean-04",
        "digital_table",
        (
            "Lipid Panel\n"
            "Total Cholesterol 228 mg/dL 0-199 H\n"
            "HDL 42 mg/dL 40-999 N\n"
            "LDL 152 mg/dL 0-99 H\n"
            "Triglycerides 210 mg/dL 0-149 H\n"
        ),
        [
            _expected("Total Cholesterol", 228.0, "mg/dL", "high"),
            _expected("HDL Cholesterol", 42.0, "mg/dL", "normal"),
            _expected("LDL Cholesterol", 152.0, "mg/dL", "high"),
            _expected("Triglycerides", 210.0, "mg/dL", "high"),
        ],
        ["high total cholesterol", "high ldl cholesterol", "high triglycerides"],
    ),
    _make_case(
        "clean-05",
        "digital_table",
        (
            "Coagulation\n"
            "PT 15.8 sec 11.0-13.5 H\n"
            "INR 1.4 0.8-1.1 H\n"
            "PTT 30.1 sec 25.0-35.0 N\n"
        ),
        [
            _expected("PT", 15.8, "sec", "high"),
            _expected("INR", 1.4, "", "high"),
            _expected("PTT", 30.1, "sec", "normal"),
        ],
        ["high pt", "high inr"],
    ),
    _make_case(
        "clean-06",
        "digital_table",
        (
            "Troponin\n"
            "Troponin I 0.12 ng/mL <0.04 H\n"
            "Troponin T 0.01 ng/mL <0.02 N\n"
        ),
        [
            _expected("Troponin I", 0.12, "ng/mL", "high"),
            _expected("Troponin T", 0.01, "ng/mL", "normal"),
        ],
        ["elevated troponin i"],
    ),
    _make_case(
        "clean-07",
        "digital_table",
        (
            "HbA1c\n"
            "HbA1c 8.2 % 4.0-5.6 H\n"
            "Glucose 185 mg/dL 70-99 H\n"
        ),
        [
            _expected("HbA1c", 8.2, "%", "high"),
            _expected("Glucose", 185.0, "mg/dL", "high"),
        ],
        ["high hba1c", "hyperglycemia"],
    ),
    _make_case(
        "clean-08",
        "digital_table",
        (
            "CBC\n"
            "WBC 3.2 K/uL 4.0-10.5 L\n"
            "Platelets 98 K/uL 150-400 L\n"
            "Hemoglobin 13.4 g/dL 12.0-15.5 N\n"
        ),
        [
            _expected("White Blood Cell Count", 3.2, "10^3/uL", "low"),
            _expected("Platelet Count", 98.0, "10^3/uL", "low"),
            _expected("Hemoglobin", 13.4, "g/dL", "normal"),
        ],
        ["leukopenia", "thrombocytopenia"],
    ),
    _make_case(
        "clean-09",
        "digital_table",
        (
            "BMP\n"
            "Sodium 146 mmol/L 135-145 H\n"
            "Potassium 2.9 mmol/L 3.5-5.1 L\n"
            "eGFR 42 60-120 L\n"
        ),
        [
            _expected("Sodium", 146.0, "mmol/L", "high"),
            _expected("Potassium", 2.9, "mmol/L", "low"),
            _expected("eGFR", 42.0, "", "low"),
        ],
        ["hypernatremia", "hypokalemia", "reduced eGFR"],
    ),
    _make_case(
        "clean-10",
        "digital_table",
        (
            "Urinalysis\n"
            "Leukocyte Esterase Positive Negative A\n"
            "Nitrite Negative Negative N\n"
            "Protein Trace Negative A\n"
        ),
        [
            _expected("Urine Leukocyte Esterase", None, "", "abnormal"),
            _expected("Urine Nitrite", None, "", "normal"),
            _expected("Urine Protein", None, "", "abnormal"),
        ],
        ["abnormal urine leukocyte esterase", "abnormal urine protein"],
    ),
]

_SCANNED_CASES = [
    _make_case(
        case["report_id"].replace("clean", "scan"),
        "image_report",
        case["content"].replace("  ", " ").replace("Range", "Range ").replace("K/uL", "K/uL"),
        case["expected_rows"],
        case["expected_findings"],
    )
    for case in _CLEAN_CASES
]

_HARD_CASES = [
    _make_case(
        "hard-01",
        "digital_table",
        (
            "panel,test,result,unit,reference range,flag\n"
            "CBC,WBC,11.8,K/uL,4.0-10.5,H\n"
            "CBC,Hemoglobin,11.1,g/dL,12.0-15.5,L\n"
        ),
        [
            _expected("White Blood Cell Count", 11.8, "10^3/uL", "high"),
            _expected("Hemoglobin", 11.1, "g/dL", "low"),
        ],
        ["leukocytosis", "low hemoglobin"],
    ),
    _make_case(
        "hard-02",
        "digital_table",
        (
            "panel\ttest\tresult\tunit\treference range\tflag\n"
            "BMP\tSodium\t129\tmmol/L\t135-145\tL\n"
            "BMP\tPotassium\t5.8\tmmol/L\t3.5-5.1\tH\n"
        ),
        [
            _expected("Sodium", 129.0, "mmol/L", "low"),
            _expected("Potassium", 5.8, "mmol/L", "high"),
        ],
        ["hyponatremia", "hyperkalemia"],
    ),
    _make_case(
        "hard-03",
        "digital_table",
        (
            "| Test | Result | Unit | Reference Range | Flag |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Creatinine | 1.8 | mg/dL | 0.6-1.3 | H |\n"
            "| eGFR | 38 |  | 60-120 | L |\n"
        ),
        [
            _expected("Creatinine", 1.8, "mg/dL", "high"),
            _expected("eGFR", 38.0, "", "low"),
        ],
        ["elevated creatinine", "reduced eGFR"],
    ),
    _make_case(
        "hard-04",
        "text_report",
        (
            "Troponin panel\n"
            "Troponin I: 0.18 ng/mL <0.04 H\n"
            "Troponin T: 0.01 ng/mL <0.02 N\n"
        ),
        [
            _expected("Troponin I", 0.18, "ng/mL", "high"),
            _expected("Troponin T", 0.01, "ng/mL", "normal"),
        ],
        ["elevated troponin i"],
    ),
    _make_case(
        "hard-05",
        "text_report",
        (
            "CMP\n"
            "Glucose 206 mg/dL 70-99 H\n"
            "BUN 31 mg/dL 7-20 H\n"
            "Creatinine 2.2 mg/dL 0.6-1.3 H\n"
        ),
        [
            _expected("Glucose", 206.0, "mg/dL", "high"),
            _expected("BUN", 31.0, "mg/dL", "high"),
            _expected("Creatinine", 2.2, "mg/dL", "high"),
        ],
        ["hyperglycemia", "high bun", "elevated creatinine"],
    ),
    _make_case(
        "hard-06",
        "text_report",
        (
            "CBC\n"
            "WBC 15.2 10^3/uL 4.0-10.5 H\n"
            "Platelets 82 10^3/uL 150-400 L\n"
        ),
        [
            _expected("White Blood Cell Count", 15.2, "10^3/uL", "high"),
            _expected("Platelet Count", 82.0, "10^3/uL", "low"),
        ],
        ["leukocytosis", "thrombocytopenia"],
    ),
    _make_case(
        "hard-07",
        "text_report",
        (
            "BMP\n"
            "Sodium 141 mmol/L 135-145 N\n"
            "Potassium 4.0 mmol/L 3.5-5.1 N\n"
            "Creatinine 1.4 mg/dL H\n"
        ),
        [
            _expected("Sodium", 141.0, "mmol/L", "normal"),
            _expected("Potassium", 4.0, "mmol/L", "normal"),
            _expected("Creatinine", 1.4, "mg/dL", "unknown"),
        ],
        [],
    ),
    _make_case(
        "hard-08",
        "text_report",
        (
            "Urinalysis\n"
            "Leukocyte Esterase Positive Negative A\n"
            "Nitrite Positive Negative A\n"
            "Ketones Negative Negative N\n"
        ),
        [
            _expected("Urine Leukocyte Esterase", None, "", "abnormal"),
            _expected("Urine Nitrite", None, "", "abnormal"),
            _expected("Urine Ketones", None, "", "normal"),
        ],
        ["abnormal urine leukocyte esterase", "abnormal urine nitrite"],
    ),
    _make_case(
        "hard-09",
        "text_report",
        (
            "HbA1c 7.9 % 4.0-5.6 H\n"
            "Glucose 165 mg/dL 70-99 H\n"
            "Potassium 3.4 mmol/L 3.5-5.1 L\n"
        ),
        [
            _expected("HbA1c", 7.9, "%", "high"),
            _expected("Glucose", 165.0, "mg/dL", "high"),
            _expected("Potassium", 3.4, "mmol/L", "low"),
        ],
        ["high hba1c", "hyperglycemia", "hypokalemia"],
    ),
    _make_case(
        "hard-10",
        "digital_table",
        (
            '[{"panel": "CBC", "test": "WBC", "result": "9.2", "unit": "K/uL", "reference range": "4.0-10.5", "flag": "N"},'
            '{"panel": "CBC", "test": "Hgb", "result": "9.8", "unit": "g/dL", "reference range": "12.0-15.5", "flag": "L"}]'
        ),
        [
            _expected("White Blood Cell Count", 9.2, "10^3/uL", "normal"),
            _expected("Hemoglobin", 9.8, "g/dL", "low"),
        ],
        ["low hemoglobin"],
    ),
]


LAB_GOLD_CASES = _CLEAN_CASES + _SCANNED_CASES + _HARD_CASES
