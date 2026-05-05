"""
No-key biomedical literature retrieval pipeline.

This module is intentionally parallel to the clinical and lab pipelines. It
does not alter the Structured Patient Profile flow; it returns a standalone
LiteratureProfile that can later be attached to a UI or downstream agent.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any

from pipeline.literature_cache import LiteratureCache
from pipeline.literature_sources import (
    enrich_with_crossref,
    fetch_europe_pmc,
    fetch_pubmed,
)
from pipeline.literature_synthesis import synthesize_literature
from schemas.literature_profile import (
    LiteratureProfile,
    LiteratureSearchRequest,
    LiteratureSourceStats,
    PaperMetadata,
)

logger = logging.getLogger(__name__)


def run_literature_pipeline(request: LiteratureSearchRequest) -> LiteratureProfile:
    """
    Retrieve, normalize, cache, deduplicate, and synthesize biomedical papers.

    The default source is Europe PMC. PubMed can be enabled by adding
    ``"pubmed"`` to ``request.sources``. Crossref acts as DOI enrichment only.
    Google Scholar scraping is intentionally unsupported.
    """
    if not request.query:
        raise ValueError("LiteratureSearchRequest.query is required.")

    warnings: list[str] = []
    manual_links: dict[str, str] = {}
    source_stats: list[LiteratureSourceStats] = []
    fetched_papers: list[PaperMetadata] = []

    if "google_scholar" in request.sources or "scholar" in request.sources:
        manual_links["google_scholar"] = build_google_scholar_url(request.query)
        warnings.append(
            "Google Scholar automated scraping is not implemented; use the manual Google Scholar link instead."
        )

    with LiteratureCache(request.cache_path or None) as cache:
        per_source_limit = _per_source_limit(request)
        source_request = _clone_request(request, max_results=per_source_limit)

        if "europe_pmc" in request.sources:
            result = fetch_europe_pmc(source_request, cache)
            fetched_papers.extend(result.papers)
            source_stats.append(
                LiteratureSourceStats(
                    source="europe_pmc",
                    requested=per_source_limit,
                    returned=result.returned,
                    cached_responses=result.cached_responses,
                    errors=result.errors,
                )
            )
            warnings.extend(result.errors)

        if "pubmed" in request.sources:
            result = fetch_pubmed(source_request, cache)
            fetched_papers.extend(result.papers)
            source_stats.append(
                LiteratureSourceStats(
                    source="pubmed",
                    requested=per_source_limit,
                    returned=result.returned,
                    cached_responses=result.cached_responses,
                    errors=result.errors,
                )
            )
            warnings.extend(result.errors)

        unsupported_sources = sorted(
            set(request.sources)
            - {"europe_pmc", "pubmed", "crossref", "google_scholar", "scholar"}
        )
        for source in unsupported_sources:
            warnings.append(f"Unsupported literature source skipped: {source}")

        papers = deduplicate_papers(fetched_papers)
        papers = sort_papers_for_display(papers)[: request.max_results]
        assign_paper_ids(papers)

        if "crossref" in request.sources:
            papers, cached_count, errors = enrich_with_crossref(papers, cache)
            source_stats.append(
                LiteratureSourceStats(
                    source="crossref",
                    requested=len([paper for paper in papers if paper.doi]),
                    returned=len([paper for paper in papers if paper.links.get("crossref")]),
                    cached_responses=cached_count,
                    errors=errors,
                )
            )
            warnings.extend(errors)

        for paper in papers:
            cache.set_paper(paper_cache_key(paper), _dump_paper(paper))

    evidence, synthesis, synthesis_warnings = synthesize_literature(request, papers)
    warnings.extend(synthesis_warnings)
    warnings.extend(_coverage_warnings(request, papers))

    return LiteratureProfile(
        request=request,
        papers=papers,
        evidence_snippets=evidence,
        synthesis=synthesis,
        source_stats=source_stats,
        warnings=_dedup(warnings),
        manual_links=manual_links,
    )


def deduplicate_papers(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """Deduplicate records across sources by DOI, PMCID, PMID, then title."""
    merged: dict[str, PaperMetadata] = {}
    order: list[str] = []

    for paper in papers:
        key = paper_cache_key(paper)
        if key not in merged:
            merged[key] = paper
            order.append(key)
            continue
        merged[key] = _merge_paper(merged[key], paper)

    return [merged[key] for key in order]


def assign_paper_ids(papers: list[PaperMetadata]) -> None:
    """Assign stable display ids P1, P2, ... in result order."""
    for index, paper in enumerate(papers, start=1):
        paper.paper_id = f"P{index}"


def paper_cache_key(paper: PaperMetadata) -> str:
    if paper.doi:
        return f"doi:{paper.doi.strip().lower()}"
    if paper.pmcid:
        return f"pmcid:{paper.pmcid.strip().lower()}"
    if paper.pmid:
        return f"pmid:{paper.pmid.strip().lower()}"
    title_key = re.sub(r"[^a-z0-9]+", " ", (paper.title or "").lower()).strip()
    return f"title:{title_key[:180]}"


def build_google_scholar_url(query: str) -> str:
    return "https://scholar.google.com/scholar?" + urllib.parse.urlencode({"q": query})


def sort_papers_for_display(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """Prioritize records that are more useful for browsing and synthesis."""
    return sorted(
        papers,
        key=lambda paper: (
            -_paper_quality_score(paper),
            -(paper.year or 0),
            paper.title.lower(),
        ),
    )


def _merge_paper(primary: PaperMetadata, secondary: PaperMetadata) -> PaperMetadata:
    primary.title = _prefer_longer(primary.title, secondary.title)
    primary.abstract = _prefer_longer(primary.abstract, secondary.abstract)
    primary.full_text = _prefer_longer(primary.full_text, secondary.full_text)
    primary.journal = primary.journal or secondary.journal
    primary.publication_date = primary.publication_date or secondary.publication_date
    primary.year = primary.year or secondary.year
    primary.doi = primary.doi or secondary.doi
    primary.pmid = primary.pmid or secondary.pmid
    primary.pmcid = primary.pmcid or secondary.pmcid
    primary.publication_type = primary.publication_type or secondary.publication_type
    primary.is_open_access = primary.is_open_access or secondary.is_open_access
    primary.has_free_full_text = primary.has_free_full_text or secondary.has_free_full_text
    primary.authors = _dedup(primary.authors + secondary.authors)
    primary.mesh_terms = _dedup(primary.mesh_terms + secondary.mesh_terms)
    primary.keywords = _dedup(primary.keywords + secondary.keywords)
    primary.links.update({key: value for key, value in secondary.links.items() if value})
    if secondary.source and secondary.source not in primary.source.split(", "):
        primary.source = ", ".join(_dedup([primary.source, secondary.source]))
    return primary


def _prefer_longer(current: str, candidate: str) -> str:
    return candidate if len(candidate or "") > len(current or "") else current


def _paper_quality_score(paper: PaperMetadata) -> int:
    score = 0
    title_lower = (paper.title or "").lower()
    if paper.abstract.strip():
        score += 5
    if paper.full_text.strip():
        score += 6
    if paper.has_free_full_text:
        score += 2
    if paper.mesh_terms:
        score += 1
    if paper.keywords:
        score += 1
    if title_lower.startswith("correction:") or title_lower.startswith("publisher correction:"):
        score -= 8
    if "plain language summary" in title_lower:
        score -= 2
    return score


def _per_source_limit(request: LiteratureSearchRequest) -> int:
    search_sources = [source for source in request.sources if source in {"europe_pmc", "pubmed"}]
    if not search_sources:
        return request.max_results
    return max(1, request.max_results)


def _clone_request(request: LiteratureSearchRequest, max_results: int) -> LiteratureSearchRequest:
    return LiteratureSearchRequest(
        query=request.query,
        theme_terms=list(request.theme_terms),
        max_results=max_results,
        year_from=request.year_from,
        year_to=request.year_to,
        publication_type=request.publication_type,
        sources=list(request.sources),
        open_access_only=request.open_access_only,
        fetch_full_text=request.fetch_full_text,
        cache_path=request.cache_path,
        email=request.email,
    )


def _coverage_warnings(request: LiteratureSearchRequest, papers: list[PaperMetadata]) -> list[str]:
    warnings: list[str] = []
    if request.open_access_only:
        warnings.append("Open-access filtering may exclude relevant paywalled papers.")
    if request.fetch_full_text:
        full_text_count = sum(bool(paper.full_text) for paper in papers)
        if full_text_count < len(papers):
            warnings.append("Full text was fetched only for records with legal free full-text XML availability.")
    return warnings


def _dump_paper(paper: PaperMetadata) -> dict[str, Any]:
    if hasattr(paper, "model_dump"):
        return paper.model_dump()
    return paper.dict()


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        key = value.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out
