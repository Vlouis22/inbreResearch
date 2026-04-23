"""
Synthetic evaluation harness for the Lab Agent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from lab_eval_cases import LAB_GOLD_CASES
from pipeline.lab_pipeline import LabInput, run_lab_pipeline


def main() -> None:
    total_expected_rows = 0
    total_predicted_rows = 0
    matched_rows = 0
    canonical_hits = 0
    numeric_hits = 0
    unit_hits = 0
    flag_hits = 0
    finding_hits = 0
    finding_total = 0

    case_summaries: list[dict] = []

    for case in LAB_GOLD_CASES:
        profile = run_lab_pipeline([LabInput(source_type=case["source_type"], content=case["content"])])
        predicted = {row.test_name_canonical or row.test_name_raw: row for row in profile.lab_results}
        expected = {row["test_name_canonical"]: row for row in case["expected_rows"]}

        expected_count = len(expected)
        predicted_count = len(predicted)
        overlap = set(expected) & set(predicted)

        total_expected_rows += expected_count
        total_predicted_rows += predicted_count
        matched_rows += len(overlap)

        for name in overlap:
            canonical_hits += 1
            expected_row = expected[name]
            predicted_row = predicted[name]

            if predicted_row.value_numeric_optional == expected_row["value_numeric_optional"]:
                numeric_hits += 1
            if (predicted_row.unit_canonical or "") == expected_row["unit_canonical"]:
                unit_hits += 1
            if predicted_row.computed_flag == expected_row["computed_flag"]:
                flag_hits += 1

        predicted_findings = {finding.lower() for finding in profile.abnormal_findings}
        expected_findings = {finding.lower() for finding in case["expected_findings"]}
        finding_hits += len(predicted_findings & expected_findings)
        finding_total += len(expected_findings)

        case_summaries.append(
            {
                "report_id": case["report_id"],
                "predicted_rows": predicted_count,
                "expected_rows": expected_count,
                "findings": profile.abnormal_findings,
                "lab_summary": profile.lab_summary,
            }
        )

    row_precision = matched_rows / total_predicted_rows if total_predicted_rows else 0.0
    row_recall = matched_rows / total_expected_rows if total_expected_rows else 0.0
    canonical_accuracy = canonical_hits / total_expected_rows if total_expected_rows else 0.0
    numeric_accuracy = numeric_hits / total_expected_rows if total_expected_rows else 0.0
    unit_accuracy = unit_hits / total_expected_rows if total_expected_rows else 0.0
    flag_accuracy = flag_hits / total_expected_rows if total_expected_rows else 0.0
    finding_recall = finding_hits / finding_total if finding_total else 1.0

    metrics = {
        "case_count": len(LAB_GOLD_CASES),
        "row_precision": round(row_precision, 4),
        "row_recall": round(row_recall, 4),
        "canonical_accuracy": round(canonical_accuracy, 4),
        "numeric_accuracy": round(numeric_accuracy, 4),
        "unit_accuracy": round(unit_accuracy, 4),
        "flag_accuracy": round(flag_accuracy, 4),
        "finding_recall": round(finding_recall, 4),
    }

    print("=" * 72)
    print("  Lab Agent — Synthetic Evaluation")
    print("=" * 72)
    print(json.dumps(metrics, indent=2))
    print("-" * 72)
    print(json.dumps(case_summaries[:5], indent=2))
    print("=" * 72)

    assert metrics["case_count"] >= 30, "Gold dataset must contain at least 30 cases."
    assert row_precision >= 0.95, f"Row precision too low: {row_precision:.3f}"
    assert row_recall >= 0.95, f"Row recall too low: {row_recall:.3f}"
    assert numeric_accuracy >= 0.95, f"Numeric accuracy too low: {numeric_accuracy:.3f}"
    assert unit_accuracy >= 0.95, f"Unit normalization accuracy too low: {unit_accuracy:.3f}"
    assert flag_accuracy >= 0.95, f"Abnormality accuracy too low: {flag_accuracy:.3f}"

    print("  ✓ Lab Agent synthetic evaluation passed.")


if __name__ == "__main__":
    main()
