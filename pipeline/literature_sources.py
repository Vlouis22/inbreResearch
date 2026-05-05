"""
Source adapters for public no-key literature retrieval.

The adapters use official APIs rather than Google Scholar scraping:
Europe PMC for biomedical search/full text, PubMed E-utilities for optional
PubMed XML, and Crossref for DOI/license enrichment.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from pipeline.literature_cache import LiteratureCache
from schemas.literature_profile import LiteratureSearchRequest, PaperMetadata

logger = logging.getLogger(__name__)

EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
USER_AGENT = "InbreResearchLiteraturePipeline/1.0 (academic no-key prototype)"
HTTP_TIMEOUT_S = 20
PUBMED_MIN_INTERVAL_S = 0.34

_last_pubmed_request_s = 0.0


@dataclass
class SourceFetchResult:
    papers: list[PaperMetadata] = field(default_factory=list)
    returned: int = 0
    cached_responses: int = 0
    errors: list[str] = field(default_factory=list)


def build_europe_pmc_query(request: LiteratureSearchRequest) -> str:
    """Build a Europe PMC query string with conservative OA/full-text filters."""
    parts = [f"({request.query})" if request.query else "*"]
    if request.open_access_only:
        parts.append("OPEN_ACCESS:y")
    if request.fetch_full_text:
        parts.append("HAS_FT:y")
    if request.year_from or request.year_to:
        start = request.year_from or 1800
        end = request.year_to or 3000
        parts.append(f"FIRST_PDATE:[{start}-01-01 TO {end}-12-31]")
    if request.publication_type:
        parts.append(f'PUB_TYPE:"{request.publication_type}"')
    return " AND ".join(parts)


def build_pubmed_query(request: LiteratureSearchRequest) -> str:
    """Build a PubMed E-utilities query string."""
    parts = [request.query or "all[sb]"]
    if request.open_access_only or request.fetch_full_text:
        parts.append("free full text[sb]")
    if request.year_from or request.year_to:
        start = request.year_from or 1800
        end = request.year_to or 3000
        parts.append(f'("{start}"[Date - Publication] : "{end}"[Date - Publication])')
    if request.publication_type:
        parts.append(f"{request.publication_type}[Publication Type]")
    return " AND ".join(parts)


def fetch_europe_pmc(request: LiteratureSearchRequest, cache: LiteratureCache) -> SourceFetchResult:
    """Fetch metadata from Europe PMC and optional free full-text XML."""
    query = build_europe_pmc_query(request)
    remaining = request.max_results
    cursor_mark = "*"
    papers: list[PaperMetadata] = []
    cached_count = 0
    errors: list[str] = []

    while remaining > 0:
        page_size = min(100, remaining)
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(page_size),
            "cursorMark": cursor_mark,
            "resultType": "core",
        }
        try:
            payload, cached = _read_url("europe_pmc", EUROPE_PMC_SEARCH_URL, params, cache)
            cached_count += int(cached)
            data = json.loads(payload)
        except Exception as exc:
            message = f"Europe PMC search failed: {exc}"
            logger.warning(message)
            errors.append(message)
            break

        page_papers = parse_europe_pmc_response(data)
        if request.fetch_full_text:
            for paper in page_papers:
                if not paper.has_free_full_text:
                    continue
                full_text, full_text_cached = fetch_europe_pmc_full_text(paper, cache)
                cached_count += int(full_text_cached)
                paper.full_text = full_text

        papers.extend(page_papers)
        remaining -= len(page_papers)
        next_cursor = str(data.get("nextCursorMark") or "")
        if not page_papers or not next_cursor or next_cursor == cursor_mark:
            break
        cursor_mark = next_cursor

    return SourceFetchResult(
        papers=papers[: request.max_results],
        returned=len(papers[: request.max_results]),
        cached_responses=cached_count,
        errors=errors,
    )


def parse_europe_pmc_response(payload: dict[str, Any] | str) -> list[PaperMetadata]:
    """Parse Europe PMC JSON search response into normalized paper records."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    results = payload.get("resultList", {}).get("result", [])
    papers: list[PaperMetadata] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "europe_pmc")
        record_id = str(item.get("id") or item.get("pmid") or "").strip()
        doi = _clean_optional(item.get("doi"))
        pmid = _clean_optional(item.get("pmid") or (record_id if source.upper() == "MED" else None))
        pmcid = _clean_optional(item.get("pmcid"))
        links = _build_literature_links(doi=doi, pmid=pmid, pmcid=pmcid)
        if record_id:
            links["europe_pmc"] = f"https://europepmc.org/article/{source}/{record_id}"

        papers.append(
            PaperMetadata(
                title=_strip_markup(str(item.get("title") or "")),
                authors=_split_authors(str(item.get("authorString") or "")),
                abstract=_strip_markup(str(item.get("abstractText") or "")),
                journal=str(item.get("journalTitle") or item.get("bookOrReportDetails") or ""),
                publication_date=str(item.get("firstPublicationDate") or item.get("journalInfo", {}).get("printPublicationDate") or ""),
                year=_parse_year(item.get("pubYear") or item.get("firstPublicationDate")),
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                source="europe_pmc",
                publication_type=str(item.get("pubType") or ""),
                is_open_access=_is_yes(item.get("isOpenAccess")),
                has_free_full_text=_is_yes(item.get("hasFullText")) or bool(pmcid),
                mesh_terms=_parse_europe_pmc_mesh(item),
                keywords=_parse_keyword_list(item.get("keywordList")),
                links=links,
            )
        )
    return papers


def fetch_europe_pmc_full_text(paper: PaperMetadata, cache: LiteratureCache) -> tuple[str, bool]:
    """Fetch free full-text XML from Europe PMC when the source exposes it."""
    source, source_id = _full_text_endpoint_for_paper(paper)
    if not source or not source_id:
        return "", False

    url = f"{EUROPE_PMC_BASE_URL}/{source}/{source_id}/fullTextXML"
    try:
        payload, cached = _read_url("europe_pmc_full_text", url, {}, cache)
    except Exception as exc:
        logger.info("Europe PMC full-text fetch skipped for %s: %s", source_id, exc)
        return "", False
    return parse_europe_pmc_full_text_xml(payload), cached


def _full_text_endpoint_for_paper(paper: PaperMetadata) -> tuple[str, str]:
    """
    Choose the best Europe PMC full-text endpoint for a paper.

    Prefer PMCID/PMC because Europe PMC fullTextXML is primarily exposed for
    PMC-hosted articles. Falling back to PMID/MED is still useful for records
    that do not expose a PMCID.
    """
    if paper.pmcid:
        return "PMC", paper.pmcid
    if paper.pmid:
        return "MED", paper.pmid
    return "", ""


def parse_europe_pmc_full_text_xml(xml_text: str) -> str:
    """Convert open full-text XML into simple section-prefixed plain text."""
    if not xml_text.strip():
        return ""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""

    paragraphs: list[str] = []
    for sec in root.findall(".//sec"):
        title_node = sec.find("./title")
        section_title = _node_text(title_node) if title_node is not None else "Full text"
        for para in sec.findall("./p"):
            text = _normalize_space(_node_text(para))
            if text:
                paragraphs.append(f"[{section_title}] {text}")

    if not paragraphs:
        for para in root.findall(".//p"):
            text = _normalize_space(_node_text(para))
            if text:
                paragraphs.append(f"[Full text] {text}")
    return "\n".join(paragraphs)


def fetch_pubmed(request: LiteratureSearchRequest, cache: LiteratureCache) -> SourceFetchResult:
    """Fetch PubMed IDs via ESearch and details via EFetch XML."""
    ids: list[str] = []
    cached_count = 0
    errors: list[str] = []
    query = build_pubmed_query(request)

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(request.max_results),
        "tool": "InbreResearch",
    }
    if request.email:
        params["email"] = request.email

    try:
        _throttle_pubmed()
        payload, cached = _read_url("pubmed_esearch", PUBMED_ESEARCH_URL, params, cache)
        cached_count += int(cached)
        data = json.loads(payload)
        ids = [str(item) for item in data.get("esearchresult", {}).get("idlist", [])]
    except Exception as exc:
        message = f"PubMed ESearch failed: {exc}"
        logger.warning(message)
        return SourceFetchResult(errors=[message])

    if not ids:
        return SourceFetchResult(returned=0, cached_responses=cached_count)

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids[: request.max_results]),
        "retmode": "xml",
        "tool": "InbreResearch",
    }
    if request.email:
        fetch_params["email"] = request.email

    try:
        _throttle_pubmed()
        xml_payload, cached = _read_url("pubmed_efetch", PUBMED_EFETCH_URL, fetch_params, cache)
        cached_count += int(cached)
        papers = parse_pubmed_xml(xml_payload)
    except Exception as exc:
        message = f"PubMed EFetch failed: {exc}"
        logger.warning(message)
        errors.append(message)
        papers = []

    return SourceFetchResult(
        papers=papers[: request.max_results],
        returned=len(papers[: request.max_results]),
        cached_responses=cached_count,
        errors=errors,
    )


def parse_pubmed_xml(xml_text: str) -> list[PaperMetadata]:
    """Parse PubMed EFetch XML into normalized paper records."""
    if not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    papers: list[PaperMetadata] = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("./MedlineCitation")
        article_node = medline.find("./Article") if medline is not None else None
        if medline is None or article_node is None:
            continue

        pmid = _clean_optional(_node_text(medline.find("./PMID")))
        title = _normalize_space(_node_text(article_node.find("./ArticleTitle")))
        abstract_parts = []
        for abstract_node in article_node.findall("./Abstract/AbstractText"):
            label = abstract_node.attrib.get("Label")
            text = _normalize_space(_node_text(abstract_node))
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)

        article_ids = article.findall("./PubmedData/ArticleIdList/ArticleId")
        doi = None
        pmcid = None
        for article_id in article_ids:
            id_type = article_id.attrib.get("IdType", "").lower()
            value = _clean_optional(_node_text(article_id))
            if id_type == "doi":
                doi = value
            elif id_type == "pmc":
                pmcid = value

        journal = _normalize_space(_node_text(article_node.find("./Journal/Title")))
        pub_date = article_node.find("./Journal/JournalIssue/PubDate")
        publication_date = _pubmed_date_text(pub_date)
        year = _parse_year(publication_date)
        links = _build_literature_links(doi=doi, pmid=pmid, pmcid=pmcid)

        papers.append(
            PaperMetadata(
                title=title,
                authors=_parse_pubmed_authors(article_node),
                abstract=" ".join(abstract_parts),
                journal=journal,
                publication_date=publication_date,
                year=year,
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                source="pubmed",
                publication_type=", ".join(_pubmed_publication_types(article_node)),
                is_open_access=bool(pmcid),
                has_free_full_text=bool(pmcid),
                mesh_terms=_parse_pubmed_mesh(medline),
                keywords=_parse_pubmed_keywords(medline),
                links=links,
            )
        )
    return papers


def enrich_with_crossref(papers: list[PaperMetadata], cache: LiteratureCache) -> tuple[list[PaperMetadata], int, list[str]]:
    """Add Crossref links/license metadata for records that already have a DOI."""
    cached_count = 0
    errors: list[str] = []
    for paper in papers:
        if not paper.doi:
            continue
        doi_path = urllib.parse.quote(paper.doi, safe="")
        url = f"{CROSSREF_WORKS_URL}/{doi_path}"
        try:
            payload, cached = _read_url("crossref", url, {}, cache)
            cached_count += int(cached)
            parse_crossref_work_response(payload, paper)
        except Exception as exc:
            message = f"Crossref enrichment failed for {paper.doi}: {exc}"
            logger.info(message)
            errors.append(message)
    return papers, cached_count, errors


def parse_crossref_work_response(payload: dict[str, Any] | str, paper: PaperMetadata) -> PaperMetadata:
    """Apply Crossref work metadata to a paper in place."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    message = payload.get("message", {})
    url = message.get("URL")
    if url:
        paper.links["crossref"] = str(url)

    licenses = message.get("license") or []
    if licenses and isinstance(licenses, list):
        first = licenses[0]
        if isinstance(first, dict) and first.get("URL"):
            paper.links["license"] = str(first["URL"])

    crossref_type = message.get("type")
    if crossref_type and not paper.publication_type:
        paper.publication_type = str(crossref_type)
    return paper


def _read_url(provider: str, url: str, params: dict[str, str], cache: LiteratureCache) -> tuple[str, bool]:
    cache_key = _cache_key(url, params)
    cached = cache.get_api_response(provider, cache_key)
    if cached is not None:
        return cached, True

    full_url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    request = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        payload = response.read().decode("utf-8", errors="replace")
    cache.set_api_response(provider, cache_key, payload)
    return payload, False


def _cache_key(url: str, params: dict[str, str]) -> str:
    if not params:
        return url
    return f"{url}?{urllib.parse.urlencode(sorted(params.items()))}"


def _throttle_pubmed() -> None:
    global _last_pubmed_request_s
    now = time.monotonic()
    elapsed = now - _last_pubmed_request_s
    if elapsed < PUBMED_MIN_INTERVAL_S:
        time.sleep(PUBMED_MIN_INTERVAL_S - elapsed)
    _last_pubmed_request_s = time.monotonic()


def _parse_europe_pmc_mesh(item: dict[str, Any]) -> list[str]:
    mesh_block = item.get("meshHeadingList") or {}
    headings = mesh_block.get("meshHeading") if isinstance(mesh_block, dict) else None
    if not isinstance(headings, list):
        return []
    terms = []
    for heading in headings:
        if isinstance(heading, dict):
            term = heading.get("descriptorName") or heading.get("term")
            if term:
                terms.append(str(term))
    return _dedup(terms)


def _parse_keyword_list(value: Any) -> list[str]:
    if isinstance(value, dict):
        keywords = value.get("keyword")
        if isinstance(keywords, list):
            return _dedup([str(item) for item in keywords])
        if isinstance(keywords, str):
            return _dedup([keywords])
    if isinstance(value, list):
        return _dedup([str(item) for item in value])
    return []


def _parse_pubmed_authors(article_node: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in article_node.findall("./AuthorList/Author"):
        collective = _node_text(author.find("./CollectiveName"))
        if collective:
            authors.append(_normalize_space(collective))
            continue
        last = _node_text(author.find("./LastName"))
        fore = _node_text(author.find("./ForeName")) or _node_text(author.find("./Initials"))
        full = _normalize_space(f"{fore} {last}".strip())
        if full:
            authors.append(full)
    return _dedup(authors)


def _pubmed_publication_types(article_node: ET.Element) -> list[str]:
    return _dedup([
        _normalize_space(_node_text(node))
        for node in article_node.findall("./PublicationTypeList/PublicationType")
        if _node_text(node)
    ])


def _parse_pubmed_mesh(medline: ET.Element) -> list[str]:
    return _dedup([
        _normalize_space(_node_text(node))
        for node in medline.findall("./MeshHeadingList/MeshHeading/DescriptorName")
        if _node_text(node)
    ])


def _parse_pubmed_keywords(medline: ET.Element) -> list[str]:
    return _dedup([
        _normalize_space(_node_text(node))
        for node in medline.findall("./KeywordList/Keyword")
        if _node_text(node)
    ])


def _pubmed_date_text(pub_date: ET.Element | None) -> str:
    if pub_date is None:
        return ""
    year = _node_text(pub_date.find("./Year"))
    month = _node_text(pub_date.find("./Month"))
    day = _node_text(pub_date.find("./Day"))
    if year:
        return "-".join(part for part in (year, month, day) if part)
    return _normalize_space(_node_text(pub_date.find("./MedlineDate")))


def _build_literature_links(doi: str | None, pmid: str | None, pmcid: str | None) -> dict[str, str]:
    links: dict[str, str] = {}
    if doi:
        links["doi"] = f"https://doi.org/{doi}"
    if pmid:
        links["pubmed"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if pmcid:
        links["pmc"] = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    return links


def _split_authors(author_string: str) -> list[str]:
    if not author_string.strip():
        return []
    parts = re.split(r"\s*,\s*|\s+;\s*", author_string)
    return _dedup([part for part in parts if part])


def _strip_markup(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return _normalize_space(text)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext())


def _parse_year(value: Any) -> int | None:
    match = re.search(r"(18|19|20|21)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _clean_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _is_yes(value: Any) -> bool:
    return str(value or "").strip().upper() in {"Y", "YES", "TRUE", "1"}


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = _normalize_space(item)
        key = value.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out
