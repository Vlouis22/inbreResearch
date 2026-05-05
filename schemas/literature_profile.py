"""
Pydantic schemas for the no-key literature retrieval pipeline.
"""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field, model_validator
    _HAS_MODEL_VALIDATOR = True
except ImportError:
    from pydantic import BaseModel, Field  # type: ignore
    _HAS_MODEL_VALIDATOR = False


def _clean_list(items: list[str] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items or []:
        value = str(item or "").strip()
        key = value.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


class PaperMetadata(BaseModel):
    """Normalized metadata for one literature record."""

    paper_id: str = Field(default="", description="Stable display id assigned by the pipeline")
    title: str = Field(default="", description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Author display names")
    abstract: str = Field(default="", description="Abstract text when available")
    journal: str = Field(default="", description="Journal or publication venue")
    publication_date: str = Field(default="", description="Publication date string")
    year: int | None = Field(default=None, description="Publication year")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    pmid: str | None = Field(default=None, description="PubMed identifier")
    pmcid: str | None = Field(default=None, description="PubMed Central identifier")
    source: str = Field(default="", description="Source adapter that returned the record")
    publication_type: str = Field(default="", description="Publication type when available")
    is_open_access: bool = Field(default=False, description="Whether the source reports open-access availability")
    has_free_full_text: bool = Field(default=False, description="Whether legal free full text is available")
    mesh_terms: list[str] = Field(default_factory=list, description="MeSH terms")
    keywords: list[str] = Field(default_factory=list, description="Author/source keywords")
    links: dict[str, str] = Field(default_factory=dict, description="Useful source links")
    full_text: str = Field(default="", description="Plain text extracted from free full-text XML")


class EvidenceSnippet(BaseModel):
    """One cited evidence sentence selected for deterministic synthesis."""

    snippet_id: str = Field(default="", description="Stable snippet id")
    paper_id: str = Field(default="", description="Paper id this evidence came from")
    section: str = Field(default="", description="Source section such as abstract/results")
    text: str = Field(default="", description="Evidence sentence")
    score: float = Field(default=0.0, description="Deterministic relevance score")
    matched_terms: list[str] = Field(default_factory=list, description="Query/theme terms found in this snippet")


class LiteratureSynthesis(BaseModel):
    """Deterministic, evidence-backed synthesis output."""

    query: str = Field(default="", description="Original literature query")
    top_themes: list[str] = Field(default_factory=list, description="Most supported query/theme terms")
    bullets: list[str] = Field(default_factory=list, description="Cited synthesis bullets")
    limitations: list[str] = Field(default_factory=list, description="Caveats about source coverage and evidence")
    paper_table: list[dict[str, str]] = Field(default_factory=list, description="Compact table-ready paper rows")


class LiteratureSourceStats(BaseModel):
    """Source-level retrieval accounting."""

    source: str = Field(default="", description="Source adapter name")
    requested: int = Field(default=0, description="Requested maximum records")
    returned: int = Field(default=0, description="Records returned before cross-source deduplication")
    cached_responses: int = Field(default=0, description="API responses served from local SQLite cache")
    errors: list[str] = Field(default_factory=list, description="Recoverable source errors")


if _HAS_MODEL_VALIDATOR:
    class LiteratureSearchRequest(BaseModel):
        """Inputs for a no-key literature search run."""

        query: str = Field(default="", description="Search text, e.g. liver cancer immunotherapy")
        theme_terms: list[str] = Field(default_factory=list, description="Specific terms to emphasize in synthesis")
        max_results: int = Field(default=100, description="Maximum retrievable records across enabled sources")
        year_from: int | None = Field(default=None, description="Inclusive start publication year")
        year_to: int | None = Field(default=None, description="Inclusive end publication year")
        publication_type: str | None = Field(default=None, description="Optional publication type filter")
        sources: list[str] = Field(default_factory=lambda: ["europe_pmc"], description="Enabled source adapters")
        open_access_only: bool = Field(default=True, description="Prefer legally free/open-access records")
        fetch_full_text: bool = Field(default=False, description="Fetch free full-text XML where available")
        cache_path: str = Field(default="", description="Optional SQLite cache path override")
        email: str | None = Field(default=None, description="Optional contact email for NCBI E-utilities")

        @model_validator(mode="after")
        def normalize_request(self) -> "LiteratureSearchRequest":
            self.query = (self.query or "").strip()
            self.theme_terms = _clean_list(self.theme_terms)
            self.sources = _clean_list(self.sources) or ["europe_pmc"]
            self.sources = [source.strip().lower().replace("-", "_") for source in self.sources]
            self.max_results = max(1, min(int(self.max_results or 1), 1000))
            if self.year_from and self.year_to and self.year_from > self.year_to:
                self.year_from, self.year_to = self.year_to, self.year_from
            if self.publication_type is not None:
                self.publication_type = self.publication_type.strip() or None
            if self.email is not None:
                self.email = self.email.strip() or None
            return self

    class LiteratureProfile(BaseModel):
        """Final output of the literature retrieval pipeline."""

        request: LiteratureSearchRequest = Field(default_factory=LiteratureSearchRequest)
        papers: list[PaperMetadata] = Field(default_factory=list)
        evidence_snippets: list[EvidenceSnippet] = Field(default_factory=list)
        synthesis: LiteratureSynthesis = Field(default_factory=LiteratureSynthesis)
        source_stats: list[LiteratureSourceStats] = Field(default_factory=list)
        warnings: list[str] = Field(default_factory=list)
        manual_links: dict[str, str] = Field(default_factory=dict)

        @model_validator(mode="before")
        @classmethod
        def coerce_none_to_empty(cls, values: Any) -> Any:
            if isinstance(values, dict):
                for field in ("papers", "evidence_snippets", "source_stats", "warnings"):
                    if values.get(field) is None:
                        values[field] = []
                if values.get("manual_links") is None:
                    values["manual_links"] = {}
            return values

else:
    class LiteratureSearchRequest(BaseModel):  # type: ignore[no-redef]
        """Fallback schema for environments without pydantic v2 validators."""

        query: str = Field(default="", description="Search text, e.g. liver cancer immunotherapy")
        theme_terms: list[str] = Field(default_factory=list, description="Specific terms to emphasize in synthesis")
        max_results: int = Field(default=100, description="Maximum retrievable records across enabled sources")
        year_from: int | None = Field(default=None, description="Inclusive start publication year")
        year_to: int | None = Field(default=None, description="Inclusive end publication year")
        publication_type: str | None = Field(default=None, description="Optional publication type filter")
        sources: list[str] = Field(default_factory=lambda: ["europe_pmc"], description="Enabled source adapters")
        open_access_only: bool = Field(default=True, description="Prefer legally free/open-access records")
        fetch_full_text: bool = Field(default=False, description="Fetch free full-text XML where available")
        cache_path: str = Field(default="", description="Optional SQLite cache path override")
        email: str | None = Field(default=None, description="Optional contact email for NCBI E-utilities")

        def __init__(self, **data: Any):
            super().__init__(**data)
            self.query = (self.query or "").strip()
            self.theme_terms = _clean_list(self.theme_terms)
            self.sources = _clean_list(self.sources) or ["europe_pmc"]
            self.sources = [source.strip().lower().replace("-", "_") for source in self.sources]
            self.max_results = max(1, min(int(self.max_results or 1), 1000))
            if self.year_from and self.year_to and self.year_from > self.year_to:
                self.year_from, self.year_to = self.year_to, self.year_from
            if self.publication_type is not None:
                self.publication_type = self.publication_type.strip() or None
            if self.email is not None:
                self.email = self.email.strip() or None

    class LiteratureProfile(BaseModel):  # type: ignore[no-redef]
        """Fallback final output schema."""

        request: LiteratureSearchRequest = Field(default_factory=LiteratureSearchRequest)
        papers: list[PaperMetadata] = Field(default_factory=list)
        evidence_snippets: list[EvidenceSnippet] = Field(default_factory=list)
        synthesis: LiteratureSynthesis = Field(default_factory=LiteratureSynthesis)
        source_stats: list[LiteratureSourceStats] = Field(default_factory=list)
        warnings: list[str] = Field(default_factory=list)
        manual_links: dict[str, str] = Field(default_factory=dict)

        def __init__(self, **data: Any):
            for field in ("papers", "evidence_snippets", "source_stats", "warnings"):
                if data.get(field) is None:
                    data[field] = []
            if data.get("manual_links") is None:
                data["manual_links"] = {}
            super().__init__(**data)
