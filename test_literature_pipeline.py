"""
Fixture-based tests for the no-key literature retrieval pipeline.
No network calls are required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pipeline import literature_pipeline as literature_pipeline_module
from pipeline.literature_exports import export_literature_papers_csv, export_literature_profile_json
from pipeline.literature_pipeline import assign_paper_ids, deduplicate_papers
from pipeline.literature_sources import (
    SourceFetchResult,
    _full_text_endpoint_for_paper,
    build_europe_pmc_query,
    build_pubmed_query,
    parse_crossref_work_response,
    parse_europe_pmc_full_text_xml,
    parse_europe_pmc_response,
    parse_pubmed_xml,
)
from pipeline.literature_synthesis import synthesize_literature
from schemas.literature_profile import LiteratureSearchRequest, PaperMetadata


EUROPE_PMC_FIXTURE = {
    "resultList": {
        "result": [
            {
                "id": "12345678",
                "source": "MED",
                "title": "Immunotherapy response patterns in liver cancer",
                "authorString": "Ada Smith, Grace Lee",
                "abstractText": (
                    "Immune checkpoint therapy in liver cancer was associated with durable response. "
                    "The results highlight biomarkers that may guide immunotherapy selection."
                ),
                "journalTitle": "Journal of Liver Oncology",
                "firstPublicationDate": "2024-03-10",
                "pubYear": "2024",
                "doi": "10.1000/liver.2024.1",
                "pmcid": "PMC123456",
                "isOpenAccess": "Y",
                "hasFullText": "Y",
                "pubType": "research article",
                "meshHeadingList": {
                    "meshHeading": [
                        {"descriptorName": "Liver Neoplasms"},
                        {"descriptorName": "Immunotherapy"},
                    ]
                },
                "keywordList": {"keyword": ["hepatocellular carcinoma", "checkpoint blockade"]},
            }
        ]
    }
}


PUBMED_XML_FIXTURE = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><Year>2024</Year><Month>Mar</Month><Day>10</Day></PubDate></JournalIssue>
          <Title>Journal of Liver Oncology</Title>
        </Journal>
        <ArticleTitle>Immunotherapy response patterns in liver cancer</ArticleTitle>
        <Abstract>
          <AbstractText Label="Results">Immune checkpoint therapy in liver cancer improved clinical response in selected patients.</AbstractText>
          <AbstractText Label="Conclusion">Biomarker testing may help select immunotherapy candidates.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><ForeName>Ada</ForeName><LastName>Smith</LastName></Author>
          <Author><ForeName>Grace</ForeName><LastName>Lee</LastName></Author>
        </AuthorList>
        <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Liver Neoplasms</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName>Immunotherapy</DescriptorName></MeshHeading>
      </MeshHeadingList>
      <KeywordList><Keyword>hepatocellular carcinoma</Keyword></KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/liver.2024.1</ArticleId>
        <ArticleId IdType="pmc">PMC123456</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


FULL_TEXT_XML_FIXTURE = """
<article>
  <body>
    <sec>
      <title>Results</title>
      <p>Patients with liver cancer receiving immunotherapy had improved response compared with historical controls.</p>
    </sec>
    <sec>
      <title>Conclusion</title>
      <p>These findings support continued study of immunotherapy biomarkers in liver malignancy.</p>
    </sec>
  </body>
</article>
"""


def main() -> None:
    request = LiteratureSearchRequest(
        query="liver cancer immunotherapy",
        theme_terms=["liver cancer", "immunotherapy"],
        max_results=25,
        year_from=2020,
        year_to=2024,
        publication_type="Journal Article",
        open_access_only=True,
        fetch_full_text=True,
    )

    europe_query = build_europe_pmc_query(request)
    assert "OPEN_ACCESS:y" in europe_query
    assert "HAS_FT:y" in europe_query
    assert "FIRST_PDATE:[2020-01-01 TO 2024-12-31]" in europe_query
    assert 'PUB_TYPE:"Journal Article"' in europe_query

    pubmed_query = build_pubmed_query(request)
    assert "free full text[sb]" in pubmed_query
    assert '"2020"[Date - Publication]' in pubmed_query
    assert "Journal Article[Publication Type]" in pubmed_query

    europe_papers = parse_europe_pmc_response(EUROPE_PMC_FIXTURE)
    assert len(europe_papers) == 1
    assert europe_papers[0].doi == "10.1000/liver.2024.1"
    assert europe_papers[0].pmid == "12345678"
    assert europe_papers[0].pmcid == "PMC123456"
    assert europe_papers[0].has_free_full_text is True
    assert "Liver Neoplasms" in europe_papers[0].mesh_terms

    pubmed_papers = parse_pubmed_xml(PUBMED_XML_FIXTURE)
    assert len(pubmed_papers) == 1
    assert pubmed_papers[0].doi == "10.1000/liver.2024.1"
    assert pubmed_papers[0].authors == ["Ada Smith", "Grace Lee"]
    assert "Biomarker testing" in pubmed_papers[0].abstract

    full_text = parse_europe_pmc_full_text_xml(FULL_TEXT_XML_FIXTURE)
    assert "[Results]" in full_text
    assert "immunotherapy had improved response" in full_text

    assert _full_text_endpoint_for_paper(europe_papers[0]) == ("PMC", "PMC123456")

    europe_papers[0].full_text = full_text
    combined = deduplicate_papers(europe_papers + pubmed_papers)
    assert len(combined) == 1
    assert combined[0].abstract.startswith("Results:")
    assign_paper_ids(combined)
    assert combined[0].paper_id == "P1"
    assert combined[0].pmid == "12345678"

    crossref_payload = {
        "message": {
            "URL": "https://doi.org/10.1000/liver.2024.1",
            "type": "journal-article",
            "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
        }
    }
    parse_crossref_work_response(crossref_payload, combined[0])
    assert combined[0].links["crossref"] == "https://doi.org/10.1000/liver.2024.1"
    assert combined[0].links["license"].startswith("https://creativecommons.org")

    snippets, synthesis, warnings = synthesize_literature(request, combined)
    assert warnings == []
    assert snippets
    assert synthesis.bullets
    assert "P1" in synthesis.bullets[0]
    assert synthesis.paper_table[0]["paper_id"] == "P1"

    empty_paper = PaperMetadata(
        paper_id="P2",
        title="Unrelated cardiology paper",
        abstract="Blood pressure management was evaluated in older adults.",
    )
    no_snippets, no_synthesis, _ = synthesize_literature(request, [empty_paper])
    assert no_snippets == []
    assert no_synthesis.bullets == []
    assert any("No evidence sentence matched" in item for item in no_synthesis.limitations)

    original_europe = literature_pipeline_module.fetch_europe_pmc
    original_pubmed = literature_pipeline_module.fetch_pubmed
    try:
        def fake_europe(fetch_request, cache):
            paper = parse_europe_pmc_response(json.dumps(EUROPE_PMC_FIXTURE))[0]
            paper.full_text = full_text
            return SourceFetchResult(papers=[paper], returned=1)

        def fake_pubmed(fetch_request, cache):
            return SourceFetchResult(papers=parse_pubmed_xml(PUBMED_XML_FIXTURE), returned=1)

        literature_pipeline_module.fetch_europe_pmc = fake_europe
        literature_pipeline_module.fetch_pubmed = fake_pubmed

        with tempfile.TemporaryDirectory() as tmpdir:
            profile = literature_pipeline_module.run_literature_pipeline(
                LiteratureSearchRequest(
                    query="liver cancer immunotherapy",
                    theme_terms=["immunotherapy"],
                    max_results=10,
                    sources=["europe_pmc", "pubmed", "google_scholar"],
                    open_access_only=True,
                    fetch_full_text=True,
                    cache_path=str(Path(tmpdir) / "lit.sqlite3"),
                )
            )
            assert len(profile.papers) == 1
            assert profile.manual_links["google_scholar"].startswith("https://scholar.google.com/scholar?")
            assert profile.evidence_snippets
            assert len(profile.source_stats) == 2

            json_path = export_literature_profile_json(profile, Path(tmpdir) / "profile.json")
            csv_path = export_literature_papers_csv(profile, Path(tmpdir) / "papers.csv")
            assert json_path.exists()
            assert csv_path.exists()
            assert "paper_id" in csv_path.read_text(encoding="utf-8")
    finally:
        literature_pipeline_module.fetch_europe_pmc = original_europe
        literature_pipeline_module.fetch_pubmed = original_pubmed

    print("  ✓ Literature pipeline tests passed.")


if __name__ == "__main__":
    main()
