"""
Structured row extraction for lab reports.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re

from schemas.lab_profile import LabResultRow

logger = logging.getLogger(__name__)

_HEADER_ALIASES = {
    "test": "test",
    "analyte": "test",
    "name": "test",
    "exam": "test",
    "result": "value",
    "value": "value",
    "observation": "value",
    "unit": "unit",
    "units": "unit",
    "reference": "reference",
    "reference range": "reference",
    "normal range": "reference",
    "range": "reference",
    "flag": "flag",
    "abn flag": "flag",
    "panel": "panel",
    "specimen": "specimen",
}

_FLAG_TOKENS = {
    "H", "L", "HH", "LL", "A", "N", "HIGH", "LOW", "NORMAL", "ABNORMAL",
    "CRITICAL", "PANIC", "POSITIVE", "NEGATIVE",
}

_PANEL_PATTERNS = (
    "cbc", "complete blood count", "bmp", "basic metabolic panel",
    "cmp", "comprehensive metabolic panel", "lft", "liver function",
    "lipid", "coagulation", "urinalysis", "troponin", "a1c", "hba1c",
)

_QUALITATIVE_TOKENS = {
    "positive", "negative", "trace", "small", "moderate", "large",
    "clear", "cloudy", "normal", "abnormal",
}

_REFERENCE_SUFFIX = re.compile(
    r"(?P<range>"
    r"(?:<=?|>=?)\s*-?\d+(?:\.\d+)?"
    r"|"
    r"-?\d+(?:\.\d+)?\s*(?:-|to)\s*-?\d+(?:\.\d+)?"
    r"|"
    r"(?:negative|positive|normal|trace)"
    r")\s*$",
    re.IGNORECASE,
)

_VALUE_PATTERN = re.compile(r"(?P<comparator><=|>=|<|>)?\s*(?P<number>-?\d+(?:\.\d+)?)$")


def parse_lab_report(raw_text: str, source_kind: str = "text_report") -> list[LabResultRow]:
    """
    Parse raw lab report text into row objects.

    The parser supports JSON, CSV/TSV, pipe tables, and free-text row layouts.
    """
    text = (raw_text or "").strip()
    if not text:
        return []

    rows = _parse_json_rows(text)
    if rows:
        return rows

    rows = _parse_delimited_rows(text)
    if rows:
        return rows

    rows = _parse_pipe_table(text)
    if rows:
        return rows

    return _parse_free_text_rows(text, source_kind)


def _parse_json_rows(text: str) -> list[LabResultRow]:
    try:
        data = json.loads(text)
    except Exception:
        return []

    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            data = data["rows"]
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    rows: list[LabResultRow] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        mapped = _map_structured_row(item)
        if mapped:
            mapped.source_page_or_row = f"json:{index}"
            rows.append(mapped)
    return rows


def _parse_delimited_rows(text: str) -> list[LabResultRow]:
    lines = [line for line in text.splitlines() if line.strip()]
    panel_hint = ""
    if len(lines) >= 2 and not any(delimiter in lines[0] for delimiter in ",\t;") and any(
        delimiter in lines[1] for delimiter in ",\t;"
    ):
        panel_hint = lines[0].strip()
        text = "\n".join(lines[1:])

    sample = "\n".join(text.splitlines()[:5])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except Exception:
        return []

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        return []
    field_map = _map_headers(reader.fieldnames)
    if "test" not in field_map.values() or "value" not in field_map.values():
        return []

    rows: list[LabResultRow] = []
    for index, item in enumerate(reader, start=1):
        mapped = _map_structured_row(item, field_map=field_map)
        if mapped:
            if panel_hint and not mapped.panel_name:
                mapped.panel_name = panel_hint
            mapped.source_page_or_row = f"row:{index}"
            rows.append(mapped)
    return rows


def _parse_pipe_table(text: str) -> list[LabResultRow]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if sum("|" in line for line in lines[:6]) < 2:
        return []

    split_rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    header = next((row for row in split_rows if len(row) >= 2), None)
    if not header:
        return []
    field_map = _map_headers(header)
    if "test" not in field_map.values() or "value" not in field_map.values():
        return []

    rows: list[LabResultRow] = []
    for index, row in enumerate(split_rows[1:], start=2):
        if len(row) != len(header) or set(row) == {"---"}:
            continue
        item = dict(zip(header, row))
        mapped = _map_structured_row(item, field_map=field_map)
        if mapped:
            mapped.source_page_or_row = f"line:{index}"
            rows.append(mapped)
    return rows


def _parse_free_text_rows(text: str, source_kind: str) -> list[LabResultRow]:
    rows: list[LabResultRow] = []
    current_panel = ""

    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = _clean_line(raw_line)
        if not line:
            continue
        if _is_header_line(line):
            continue
        if _is_panel_heading(line):
            current_panel = line
            continue

        row = _parse_line_row(line, current_panel=current_panel, source_ref=f"line:{index}")
        if row:
            rows.append(row)

    logger.info("Parsed %d lab row(s) from %s text.", len(rows), source_kind)
    return rows


def _map_structured_row(item: dict, field_map: dict[str, str] | None = None) -> LabResultRow | None:
    source = field_map or _map_headers(item.keys())

    def get_field(name: str) -> str:
        for raw_key, mapped in source.items():
            if mapped == name:
                value = item.get(raw_key, "")
                return str(value).strip()
        return ""

    test_name = get_field("test")
    value = get_field("value")
    if not test_name or not value:
        return None

    return LabResultRow(
        panel_name=get_field("panel"),
        test_name_raw=test_name,
        value_raw=value,
        unit_raw=get_field("unit"),
        reference_range_raw=get_field("reference"),
        reported_flag_optional=get_field("flag") or None,
        specimen_optional=get_field("specimen") or None,
        confidence=0.96,
        source_page_or_row="",
    )


def _map_headers(headers: list[str] | object) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for header in headers:
        name = str(header).strip().lower()
        if not name:
            continue
        mapped[str(header)] = _HEADER_ALIASES.get(name, name)
    return mapped


def _clean_line(line: str) -> str:
    line = re.sub(r"[·•]", " ", line)
    line = re.sub(r"\s+", " ", line.replace("—", "-")).strip()
    return line


def _is_header_line(line: str) -> bool:
    lowered = line.lower()
    return (
        ("test" in lowered and "result" in lowered)
        or ("analyte" in lowered and "value" in lowered)
        or ("reference" in lowered and ("range" in lowered or "flag" in lowered))
    )


def _is_panel_heading(line: str) -> bool:
    lowered = line.lower()
    if re.search(r"\s(?:<=|>=|<|>)?\s*-?\d+(?:\.\d+)?", line):
        return False
    return any(pattern in lowered for pattern in _PANEL_PATTERNS)


def _parse_line_row(line: str, current_panel: str, source_ref: str) -> LabResultRow | None:
    working = line.strip().strip(":")

    reported_flag = None
    last_token = working.split()[-1].upper()
    if last_token in _FLAG_TOKENS:
        reported_flag = last_token
        working = working[: -len(working.split()[-1])].strip()

    reference_range_raw = ""
    range_match = _REFERENCE_SUFFIX.search(working)
    if range_match:
        reference_range_raw = range_match.group("range").strip()
        working = working[:range_match.start()].strip(" -:")

    tokens = working.split()
    for index, token in enumerate(tokens):
        clean_token = token.strip(",;")
        if not _VALUE_PATTERN.fullmatch(clean_token):
            continue
        comparator_match = re.match(r"(<=|>=|<|>)", clean_token)
        comparator = comparator_match.group(1) if comparator_match else None
        value_raw = clean_token[len(comparator):].strip() if comparator else clean_token
        test_name_raw = " ".join(tokens[:index]).strip(" :-")
        unit_raw = " ".join(tokens[index + 1:]).strip(" :-")
        if not test_name_raw:
            continue
        confidence = 0.90 if reference_range_raw else 0.82
        return LabResultRow(
            panel_name=current_panel,
            test_name_raw=test_name_raw,
            value_raw=value_raw,
            comparator_optional=comparator,
            unit_raw=unit_raw,
            reference_range_raw=reference_range_raw,
            reported_flag_optional=reported_flag,
            confidence=confidence,
            source_page_or_row=source_ref,
        )

    qualitative_match = re.search(r"(?P<qualitative>positive|negative|trace|small|moderate|large|normal|abnormal)$", working, re.IGNORECASE)
    if qualitative_match:
        value_raw = qualitative_match.group("qualitative")
        test_name_raw = working[:qualitative_match.start()].strip(" :-")
        if not test_name_raw:
            return None
        return LabResultRow(
            panel_name=current_panel,
            test_name_raw=test_name_raw,
            value_raw=value_raw,
            reference_range_raw=reference_range_raw,
            reported_flag_optional=reported_flag,
            confidence=0.76,
            source_page_or_row=source_ref,
        )

    return None
