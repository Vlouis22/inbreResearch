"""
Export helpers for literature profiles.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from schemas.literature_profile import LiteratureProfile


def export_literature_profile_json(profile: LiteratureProfile, path: str | Path) -> Path:
    """Write a full literature profile as formatted JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_model_dump(profile), indent=2), encoding="utf-8")
    return output_path


def export_literature_papers_csv(profile: LiteratureProfile, path: str | Path) -> Path:
    """Write normalized paper metadata as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = profile.synthesis.paper_table
    fieldnames = ["paper_id", "year", "title", "journal", "doi", "pmid", "pmcid", "source", "free_full_text"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    raise TypeError(f"Cannot serialize object of type {type(model)!r}")
