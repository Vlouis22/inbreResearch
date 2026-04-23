"""
Input routing helpers for the Lab Agent.
"""

from __future__ import annotations

from pathlib import Path


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_TABLE_SUFFIXES = {".csv", ".tsv", ".json", ".xlsx"}
_TEXT_SUFFIXES = {".txt", ".md"}
_KNOWN_SOURCE_TYPES = {"pdf_report", "image_report", "digital_table", "text_report"}


def detect_lab_input_kind(content: str, source_type: str | None = None) -> str:
    """
    Route lab content to the correct parsing path.

    `source_type` wins when it matches a supported format. This makes tests and
    demos deterministic even when the content is already OCR text.
    """
    if source_type in _KNOWN_SOURCE_TYPES:
        return source_type

    path = Path(content)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "pdf_report"
        if suffix in _IMAGE_SUFFIXES:
            return "image_report"
        if suffix in _TABLE_SUFFIXES:
            return "digital_table"
        if suffix in _TEXT_SUFFIXES:
            return "text_report"

    if _looks_like_table(content):
        return "digital_table"
    return "text_report"


def _looks_like_table(content: str) -> bool:
    lines = [line for line in (content or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    if sum("|" in line for line in lines[:5]) >= 2:
        return True
    if sum("\t" in line for line in lines[:5]) >= 2:
        return True
    if sum("," in line for line in lines[:5]) >= 2:
        return True
    if any("reference" in line.lower() and "result" in line.lower() for line in lines[:3]):
        return True
    return False
