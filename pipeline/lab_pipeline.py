"""
Lab Agent pipeline orchestration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from models.lab_document_model import extract_lab_document, load_lab_document_client
from pipeline.lab_abnormality import classify_lab_rows
from pipeline.lab_input_router import detect_lab_input_kind
from pipeline.lab_normalizer import normalize_lab_rows
from pipeline.lab_row_parser import parse_lab_report
from schemas.lab_profile import LabProfile, LabResultRow, LabSourceMetadata

logger = logging.getLogger(__name__)


@dataclass
class LabInput:
    """Represents one laboratory data source."""

    source_type: str
    content: str


def load_lab_models() -> None:
    """Prepare optional remote OCR dependencies for the Lab Agent."""
    load_lab_document_client()


def run_lab_pipeline(inputs: list[LabInput]) -> LabProfile:
    """
    Execute the full Lab Agent pipeline.

    The implementation is API-first for OCR, but it remains usable with plain
    text and digital tables when OCR is unavailable.
    """
    if not inputs:
        return LabProfile()

    raw_rows: list[LabResultRow] = []
    metadata = LabSourceMetadata()
    providers: list[str] = []

    for item in inputs:
        source_kind = detect_lab_input_kind(item.content, source_type=item.source_type)
        extraction = extract_lab_document(item.content, source_kind)
        parsed_rows = parse_lab_report(extraction.text, source_kind=source_kind)
        raw_rows.extend(parsed_rows)

        if source_kind == "pdf_report":
            metadata.pdf_report = True
        elif source_kind == "image_report":
            metadata.image_report = True
        elif source_kind == "digital_table":
            metadata.digital_table = True
        else:
            metadata.text_report = True

        metadata.fallback_parser_used = metadata.fallback_parser_used or extraction.used_fallback
        if extraction.provider and extraction.provider not in providers:
            providers.append(extraction.provider)

    normalized_rows = normalize_lab_rows(raw_rows)
    deduplicated_rows = _deduplicate_rows(normalized_rows)
    enriched_rows, findings, summary = classify_lab_rows(deduplicated_rows)

    metadata.ocr_provider = ", ".join(providers)
    logger.info("Lab pipeline complete with %d structured result(s).", len(enriched_rows))
    return LabProfile(
        lab_results=enriched_rows,
        abnormal_findings=findings,
        lab_summary=summary,
        source_metadata=metadata,
    )


def _deduplicate_rows(rows: list[LabResultRow]) -> list[LabResultRow]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[LabResultRow] = []

    for row in rows:
        key = (
            (row.panel_name or "").strip().lower(),
            (row.test_name_canonical or row.test_name_raw).strip().lower(),
            (row.value_raw or "").strip().lower(),
            (row.unit_canonical or row.unit_raw).strip().lower(),
            (row.reference_range_raw or "").strip().lower(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped
