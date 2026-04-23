"""
Lab document extraction wrapper.

Primary paths:
  • Local EasyOCR for image-based lab reports when available.
  • Hugging Face Inference Providers via `huggingface_hub.InferenceClient`
    with the configured OCR/document model.

Fallback paths:
  • Use pre-extracted text directly for tests and demos.
  • Extract text from text-based PDFs if `pypdf` is available.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path

from config import HF_INFERENCE_TIMEOUT_S, LAB_OCR_FALLBACK_MODEL, LAB_OCR_PRIMARY_MODEL

logger = logging.getLogger(__name__)


@dataclass
class DocumentExtraction:
    """Normalized output from the lab document extraction stage."""

    text: str
    confidence: float
    provider: str
    used_fallback: bool


_hf_client = None
_hf_client_loaded = False
_easyocr_reader = None
_easyocr_reader_loaded = False


def _get_hf_token() -> str | None:
    """Resolve HF_TOKEN from the environment, then fall back to the repo .env file."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return None

    try:
        for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != "HF_TOKEN":
                continue
            value = value.strip().strip("'").strip('"')
            return value or None
    except Exception as exc:
        logger.warning("Could not read HF_TOKEN from %s: %s", env_path, exc)

    return None


def load_lab_document_client() -> None:
    """Lazily load the Hugging Face inference client when available."""
    global _hf_client, _hf_client_loaded
    if _hf_client_loaded:
        return
    _hf_client_loaded = True

    token = _get_hf_token()
    if not token:
        logger.info("HF_TOKEN not found in environment or repo .env; Lab OCR will use local fallbacks until configured.")
        return

    try:
        from huggingface_hub import InferenceClient

        _hf_client = InferenceClient(
            api_key=token,
            timeout=HF_INFERENCE_TIMEOUT_S,
        )
        logger.info("Hugging Face inference client ready for Lab OCR.")
    except Exception as exc:
        logger.warning(
            "Could not initialize Hugging Face inference client. "
            "Lab OCR will use local fallbacks. Error: %s",
            exc,
        )
        _hf_client = None


def _load_easyocr_reader():
    """Lazily load EasyOCR for local image fallback when available."""
    global _easyocr_reader, _easyocr_reader_loaded
    if _easyocr_reader_loaded:
        return _easyocr_reader
    _easyocr_reader_loaded = True

    try:
        import easyocr

        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR reader ready for local Lab OCR fallback.")
    except Exception as exc:
        logger.warning("Could not initialize EasyOCR fallback: %s", exc)
        _easyocr_reader = None

    return _easyocr_reader


def extract_lab_document(
    source: str,
    source_kind: str,
    primary_model: str = LAB_OCR_PRIMARY_MODEL,
    fallback_model: str = LAB_OCR_FALLBACK_MODEL,
) -> DocumentExtraction:
    """
    Return extracted document text for a lab input.

    The function prefers local OCR for images when available, while still
    supporting remote Hugging Face OCR and pre-extracted OCR text for tests.
    """
    load_lab_document_client()

    path = Path(source)
    if source_kind in {"digital_table", "text_report"}:
        if path.exists():
            return DocumentExtraction(
                text=_read_text_file(path),
                confidence=0.98,
                provider="local-file",
                used_fallback=False,
            )
        return DocumentExtraction(
            text=source,
            confidence=0.96,
            provider="inline-text",
            used_fallback=False,
        )

    if source_kind == "pdf_report":
        if path.exists():
            text = _extract_text_from_pdf(path)
            if text.strip():
                return DocumentExtraction(
                    text=text,
                    confidence=0.88,
                    provider="pypdf",
                    used_fallback=False,
                )
        else:
            return DocumentExtraction(
                text=source,
                confidence=0.82,
                provider="pre-extracted-pdf-text",
                used_fallback=False,
            )

    if source_kind == "image_report":
        if path.exists():
            local = _extract_with_easyocr(path)
            if local.text.strip():
                return local
        if path.exists() and _hf_client is not None:
            remote = _extract_with_hf_ocr(path, primary_model)
            if remote.text.strip():
                return remote
            backup = _extract_with_hf_ocr(path, fallback_model)
            if backup.text.strip():
                return backup
        if path.exists():
            logger.warning(
                "Falling back to placeholder extraction for image report %s. "
                "Provide HF_TOKEN for OCR-backed extraction.",
                source,
            )
            return DocumentExtraction(
                text="[LAB OCR PLACEHOLDER] Unable to OCR image report locally. "
                     "Provide extracted text, install EasyOCR, or configure HF_TOKEN for OCR.",
                confidence=0.10,
                provider="placeholder",
                used_fallback=True,
            )
        return DocumentExtraction(
            text=source,
            confidence=0.80,
            provider="pre-extracted-ocr-text",
            used_fallback=False,
        )

    return DocumentExtraction(
        text=source,
        confidence=0.70,
        provider="unknown",
        used_fallback=True,
    )


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(page.strip() for page in pages if page.strip())
        if text:
            logger.info("Extracted %d characters from PDF %s via pypdf.", len(text), path.name)
        return text
    except Exception as exc:
        logger.warning("Could not extract text from PDF %s: %s", path, exc)
        return ""


def _extract_with_hf_ocr(path: Path, model_name: str) -> DocumentExtraction:
    assert _hf_client is not None

    try:
        data_url = _image_file_to_data_url(path)
        prompt = (
            "Extract this laboratory report faithfully. Preserve panel headings and "
            "emit one result per line in the order it appears. Include test name, "
            "value, unit, reference range, and flag when present. Do not infer or summarize."
        )
        response = _hf_client.chat_completion(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=2048,
        )
        text = response.choices[0].message.content if response and response.choices else ""
        if isinstance(text, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in text
            )
        text = str(text or "").strip()
        logger.info("Lab OCR completed with %s for %s.", model_name, path.name)
        return DocumentExtraction(
            text=text,
            confidence=0.78,
            provider=model_name,
            used_fallback=False,
        )
    except Exception as exc:
        logger.warning("Lab OCR failed with %s on %s: %s", model_name, path, exc)
        return DocumentExtraction(
            text="",
            confidence=0.0,
            provider=model_name,
            used_fallback=True,
        )


def _extract_with_easyocr(path: Path) -> DocumentExtraction:
    reader = _load_easyocr_reader()
    if reader is None:
        return DocumentExtraction(
            text="",
            confidence=0.0,
            provider="easyocr",
            used_fallback=True,
        )

    try:
        results = reader.readtext(str(path), detail=1, paragraph=False)
        text = _easyocr_results_to_tsv(results)
        if text.strip():
            logger.info("Lab OCR completed with EasyOCR for %s.", path.name)
            return DocumentExtraction(
                text=text,
                confidence=0.74,
                provider="easyocr",
                used_fallback=True,
            )
    except Exception as exc:
        logger.warning("EasyOCR failed on %s: %s", path, exc)

    return DocumentExtraction(
        text="",
        confidence=0.0,
        provider="easyocr",
        used_fallback=True,
    )


def _easyocr_results_to_tsv(results: list[object]) -> str:
    entries = []
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        bbox, text = item[0], str(item[1]).strip()
        if not text or not isinstance(bbox, (list, tuple)):
            continue

        xs = [float(point[0]) for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]
        ys = [float(point[1]) for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not xs or not ys:
            continue

        entries.append(
            {
                "text": text,
                "left": min(xs),
                "right": max(xs),
                "cx": (min(xs) + max(xs)) / 2.0,
                "cy": (min(ys) + max(ys)) / 2.0,
                "height": max(ys) - min(ys),
            }
        )

    if not entries:
        return ""

    entries.sort(key=lambda entry: (entry["cy"], entry["left"]))
    heights = sorted(entry["height"] for entry in entries)
    median_height = heights[len(heights) // 2] if heights else 16.0
    row_tolerance = max(12.0, median_height * 0.8)

    grouped_rows: list[dict[str, object]] = []
    for entry in entries:
        if not grouped_rows or abs(entry["cy"] - grouped_rows[-1]["cy"]) > row_tolerance:
            grouped_rows.append({"cy": entry["cy"], "entries": [entry]})
            continue

        row_entries = grouped_rows[-1]["entries"]
        count = len(row_entries)
        grouped_rows[-1]["cy"] = (grouped_rows[-1]["cy"] * count + entry["cy"]) / (count + 1)
        row_entries.append(entry)

    for row in grouped_rows:
        row["entries"].sort(key=lambda entry: entry["left"])

    header_entries = max(grouped_rows[:3], key=lambda row: len(row["entries"]))["entries"]
    column_centers = sorted(entry["cx"] for entry in header_entries)
    if len(column_centers) < 2:
        return "\n".join(entry["text"] for entry in entries)

    matrix: list[list[str]] = []
    for row in grouped_rows:
        cells = [""] * len(column_centers)
        for entry in row["entries"]:
            index = min(
                range(len(column_centers)),
                key=lambda idx: abs(entry["cx"] - column_centers[idx]),
            )
            cells[index] = f"{cells[index]} {entry['text']}".strip()
        matrix.append(cells)

    merged_rows: list[list[str]] = []
    for row in matrix:
        if (
            merged_rows
            and len(row) >= 3
            and not row[0]
            and not row[1]
            and row[2]
        ):
            merged_rows[-1][2] = f"{merged_rows[-1][2]} {row[2]}".strip()
            continue
        merged_rows.append(row)

    lines = ["Test\tResult\tUnit\tReference Range"]
    for row in merged_rows[1:]:
        test_raw, value_raw, reference_raw = (row + ["", "", ""])[:3]
        test_name, unit_from_test = _split_test_and_unit(test_raw)
        if not test_name or not value_raw:
            continue
        unit = unit_from_test or _extract_unit_from_reference(reference_raw)
        lines.append(
            "\t".join(
                [
                    _clean_cell_text(test_name),
                    _clean_cell_text(value_raw),
                    _clean_cell_text(unit),
                    _clean_cell_text(reference_raw),
                ]
            )
        )

    return "\n".join(lines)


def _split_test_and_unit(test_raw: str) -> tuple[str, str]:
    if "," in test_raw:
        test_name, maybe_unit = test_raw.rsplit(",", 1)
        if re.search(r"[A-Za-z%/µμ]", maybe_unit):
            return test_name.strip(), _clean_ocr_unit(maybe_unit)
    return test_raw.strip(), ""


def _extract_unit_from_reference(reference_raw: str) -> str:
    lowered = reference_raw.strip().lower()
    if lowered in {"", "negative", "positive", "nil", "normal", "abnormal"}:
        return ""

    for token in reversed(reference_raw.split()):
        cleaned = token.strip(".,;:()")
        if re.search(r"[A-Za-z%/µμ]", cleaned):
            return _clean_ocr_unit(cleaned)
    return ""


def _clean_ocr_unit(unit_raw: str) -> str:
    normalized = unit_raw.strip().replace("µ", "u").replace("μ", "u")
    replacements = {
        "gmldl": "gm/dl",
        "mgldl": "mg/dl",
        "mcgldl": "mcg/dl",
        "cmlmm": "cm/mm",
        "mmlhr": "mm/hr",
        "ulml": "u/ml",
    }
    return replacements.get(normalized.lower(), normalized)


def _clean_cell_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").replace("\t", " ").strip()


def _image_file_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
