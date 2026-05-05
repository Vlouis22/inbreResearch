"""
Minimal local preview app for the literature pipeline.

This is a small server-side rendered interface intended for Codex preview.
It avoids extra dependencies and exposes the no-key literature pipeline over
plain HTTP on localhost.
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from pipeline.literature_pipeline import run_literature_pipeline
from schemas.literature_profile import LiteratureSearchRequest

HOST = "127.0.0.1"
PORT = 8765

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PreviewHandler(BaseHTTPRequestHandler):
    """Simple HTML interface for running literature searches."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/download":
            self._handle_download(parsed)
            return
        if parsed.path not in {"/", "/index.html"}:
            self.send_error(404, "Not found")
            return

        params = parse_qs(parsed.query)
        query = (params.get("query", [""])[0] or "").strip()
        request = _build_request_from_params(params) if query else None
        theme_terms = (params.get("theme_terms", [""])[0] or "").strip()
        sources = params.get("sources", ["europe_pmc"])
        max_results = _safe_int(params.get("max_results", ["10"])[0], default=10)
        fetch_full_text = "fetch_full_text" in params
        open_access_only = "open_access_only" in params or not query
        year_from = _safe_optional_int(params.get("year_from", [""])[0])
        year_to = _safe_optional_int(params.get("year_to", [""])[0])

        profile_json = ""
        rendered_results = "<p class='placeholder'>Enter a topic to run the literature pipeline.</p>"
        download_controls = ""

        if request is not None:
            try:
                profile = run_literature_pipeline(request)
                profile_json = json.dumps(profile.model_dump(), indent=2)
                download_controls = _render_download_controls(request)
                rendered_results = _render_profile(profile, download_controls)
            except Exception as exc:
                logger.exception("Preview request failed.")
                rendered_results = f"<div class='error'>Pipeline error: {html.escape(str(exc))}</div>"

        page = _render_page(
            query=query,
            theme_terms=theme_terms,
            max_results=max_results,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
            fetch_full_text=fetch_full_text,
            rendered_results=rendered_results,
            profile_json=profile_json,
            sources=sources,
        )
        self._send_html(page)

    def _handle_download(self, parsed) -> None:
        params = parse_qs(parsed.query)
        download_format = (params.get("format", ["csv"])[0] or "csv").strip().lower()
        request = _build_request_from_params(params)
        if not request.query:
            self.send_error(400, "Query is required for download")
            return

        try:
            profile = run_literature_pipeline(request)
        except Exception as exc:
            logger.exception("Download request failed.")
            self.send_error(500, f"Pipeline error: {exc}")
            return

        stem = _download_stem(request.query)
        if download_format == "json":
            payload = json.dumps(profile.model_dump(), indent=2).encode("utf-8")
            self._send_bytes(payload, "application/json; charset=utf-8", f"{stem}.json")
            return
        if download_format == "csv":
            payload = _build_papers_csv(profile).encode("utf-8")
            self._send_bytes(payload, "text/csv; charset=utf-8", f"{stem}.csv")
            return
        self.send_error(400, "Unsupported download format")

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_bytes(self, payload: bytes, content_type: str, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _render_profile(profile, download_controls: str = "") -> str:
    abstract_count = sum(bool((paper.abstract or "").strip()) for paper in (profile.papers or []))
    full_text_count = sum(bool((paper.full_text or "").strip()) for paper in (profile.papers or []))
    correction_count = sum(_looks_like_notice(paper) for paper in (profile.papers or []))
    bullets = "".join(
        f"<li>{html.escape(bullet)}</li>"
        for bullet in (profile.synthesis.bullets or [])
    ) or "<li>No synthesis bullets were produced.</li>"

    themes = "".join(
        f"<span class='chip'>{html.escape(theme)}</span>"
        for theme in (profile.synthesis.top_themes or [])
    ) or "<span class='muted'>No themes</span>"

    warnings = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in (profile.warnings or [])
    ) or "<li>No warnings.</li>"

    evidence = "".join(
        (
            "<article class='evidence'>"
            f"<div class='meta'>{html.escape(snippet.snippet_id)} · {html.escape(snippet.paper_id)} · {html.escape(snippet.section)}</div>"
            f"<p>{html.escape(snippet.text)}</p>"
            "</article>"
        )
        for snippet in (profile.evidence_snippets or [])[:8]
    ) or "<p class='placeholder'>No evidence snippets selected.</p>"

    papers = "".join(_render_paper_card(paper) for paper in (profile.papers or []))
    papers = papers or "<p class='placeholder'>No papers found.</p>"

    return f"""
    <section class="band stats">
      <div class="stat"><strong>{len(profile.papers)}</strong><span>papers</span></div>
      <div class="stat"><strong>{abstract_count}</strong><span>with abstract</span></div>
      <div class="stat"><strong>{full_text_count}</strong><span>with full text</span></div>
      <div class="stat"><strong>{correction_count}</strong><span>corrections or notices</span></div>
    </section>
    {download_controls}
    <section class="band">
      <div class="section-head">
        <h2>Top themes</h2>
        <div class="chips">{themes}</div>
      </div>
      <ul class="bullets">{bullets}</ul>
    </section>
    <section class="band grid">
      <div>
        <h2>Evidence</h2>
        {evidence}
      </div>
      <div>
        <h2>Warnings</h2>
        <ul class="warnings">{warnings}</ul>
      </div>
    </section>
    <section class="band">
      <h2>Papers</h2>
      <div class="papers">{papers}</div>
    </section>
    """


def _render_paper_card(paper) -> str:
    link_parts = []
    for label, url in paper.links.items():
        link_parts.append(
            f"<a href='{html.escape(url, quote=True)}' target='_blank' rel='noreferrer'>{html.escape(label)}</a>"
        )
    links = " ".join(link_parts) or "<span class='muted'>No links</span>"
    authors = ", ".join(paper.authors[:5]) if paper.authors else "Unknown authors"
    summary = _paper_summary_text(paper)
    badges = []
    if paper.abstract.strip():
        badges.append("<span class='badge'>abstract</span>")
    if paper.full_text.strip():
        badges.append("<span class='badge'>full text</span>")
    elif paper.has_free_full_text:
        badges.append("<span class='badge badge-muted'>free full text flagged</span>")
    if _looks_like_notice(paper):
        badges.append("<span class='badge badge-warn'>notice</span>")
    badge_html = "".join(badges) or "<span class='muted'>metadata only</span>"
    return f"""
    <article class="paper">
      <div class="paper-top">
        <span class="paper-id">{html.escape(paper.paper_id)}</span>
        <span class="paper-year">{html.escape(str(paper.year or ''))}</span>
      </div>
      <h3>{html.escape(paper.title)}</h3>
      <p class="meta">{html.escape(authors)} · {html.escape(paper.journal or 'Unknown venue')}</p>
      <div class="chips compact">{badge_html}</div>
      <p>{html.escape(summary)}</p>
      <div class="links">{links}</div>
    </article>
    """


def _paper_summary_text(paper) -> str:
    abstract = (paper.abstract or "").strip()
    if abstract:
        return abstract
    full_text_snippet = _first_full_text_sentence(paper.full_text)
    if full_text_snippet:
        return f"No abstract returned by source metadata. Preview from full text: {full_text_snippet}"
    if _looks_like_notice(paper):
        return "This record appears to be a correction, notice, or related metadata entry, and the source did not provide an abstract."
    return "No abstract returned by source metadata for this record."


def _first_full_text_sentence(full_text: str) -> str:
    for raw_line in (full_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            line = line.split("]", 1)[1].strip()
        if not line:
            continue
        sentence = line.split(". ", 1)[0].strip()
        if sentence and not sentence.endswith("."):
            sentence += "."
        if len(sentence) >= 40:
            return sentence
    return ""


def _looks_like_notice(paper) -> bool:
    title = (paper.title or "").strip().lower()
    return (
        title.startswith("correction:")
        or title.startswith("publisher correction:")
        or title.startswith("erratum:")
    )


def _render_download_controls(request: LiteratureSearchRequest) -> str:
    csv_url = _build_download_url(request, "csv")
    json_url = _build_download_url(request, "json")
    return f"""
    <section class="band downloads">
      <div class="section-head">
        <h2>Downloads</h2>
        <p class="muted">Export the current result set to your computer.</p>
      </div>
      <div class="download-actions">
        <a class="button-link" href="{html.escape(csv_url, quote=True)}">Download CSV</a>
        <a class="button-link button-link-secondary" href="{html.escape(json_url, quote=True)}">Download JSON</a>
      </div>
    </section>
    """


def _build_download_url(request: LiteratureSearchRequest, download_format: str) -> str:
    params: list[tuple[str, str]] = [
        ("format", download_format),
        ("query", request.query),
        ("theme_terms", ", ".join(request.theme_terms)),
        ("max_results", str(request.max_results)),
    ]
    if request.year_from is not None:
        params.append(("year_from", str(request.year_from)))
    if request.year_to is not None:
        params.append(("year_to", str(request.year_to)))
    if request.open_access_only:
        params.append(("open_access_only", "on"))
    if request.fetch_full_text:
        params.append(("fetch_full_text", "on"))
    for source in request.sources:
        params.append(("sources", source))
    return "/download?" + urlencode(params)


def _build_request_from_params(params: dict[str, list[str]]) -> LiteratureSearchRequest:
    query = (params.get("query", [""])[0] or "").strip()
    theme_terms = (params.get("theme_terms", [""])[0] or "").strip()
    sources = params.get("sources", ["europe_pmc"])
    max_results = _safe_int(params.get("max_results", ["10"])[0], default=10)
    fetch_full_text = "fetch_full_text" in params
    open_access_only = "open_access_only" in params or not query
    year_from = _safe_optional_int(params.get("year_from", [""])[0])
    year_to = _safe_optional_int(params.get("year_to", [""])[0])
    return LiteratureSearchRequest(
        query=query,
        theme_terms=[item.strip() for item in theme_terms.split(",") if item.strip()],
        max_results=max_results,
        year_from=year_from,
        year_to=year_to,
        sources=sources,
        open_access_only=open_access_only,
        fetch_full_text=fetch_full_text,
    )


def _build_papers_csv(profile) -> str:
    fieldnames = ["paper_id", "year", "title", "journal", "doi", "pmid", "pmcid", "source", "free_full_text"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in profile.synthesis.paper_table:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return buffer.getvalue()


def _download_stem(query: str) -> str:
    compact = "-".join(part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in query).split() if part)
    compact = compact[:60] or "literature-results"
    return f"{compact}-results"


def _render_page(
    query: str,
    theme_terms: str,
    max_results: int,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
    fetch_full_text: bool,
    rendered_results: str,
    profile_json: str,
    sources: list[str],
) -> str:
    checked = "checked" if open_access_only else ""
    checked_ft = "checked" if fetch_full_text else ""
    europe_checked = "checked" if "europe_pmc" in sources else ""
    pubmed_checked = "checked" if "pubmed" in sources else ""
    scholar_checked = "checked" if "google_scholar" in sources else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Literature Preview</title>
  <style>
    :root {{
      --bg: #f3efe5;
      --panel: #fffaf0;
      --ink: #18201f;
      --muted: #5d665f;
      --line: #d7cfbf;
      --accent: #0d6b5f;
      --accent-2: #c75d2c;
      --shadow: 0 18px 50px rgba(24, 32, 31, 0.08);
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(199, 93, 44, 0.12), transparent 32%),
        radial-gradient(circle at right 20%, rgba(13, 107, 95, 0.12), transparent 28%),
        linear-gradient(180deg, #f9f5ec 0%, var(--bg) 100%);
    }}
    .shell {{
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 56px;
    }}
    .hero, .band {{
      background: color-mix(in srgb, var(--panel) 92%, white 8%);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 24px;
      border-radius: 8px;
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1;
      max-width: 10ch;
    }}
    .lede {{
      margin-top: 10px;
      color: var(--muted);
      max-width: 70ch;
      font-size: 1rem;
      line-height: 1.5;
    }}
    form {{
      margin-top: 22px;
      display: grid;
      gap: 12px;
    }}
    .row {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(12, minmax(0, 1fr));
    }}
    .field {{
      display: grid;
      gap: 6px;
    }}
    .field label, .fieldset legend {{
      font-size: 0.85rem;
      color: var(--muted);
    }}
    .field input[type="text"], .field input[type="number"] {{
      width: 100%;
      min-height: 46px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
    }}
    .query {{ grid-column: span 6; }}
    .themes {{ grid-column: span 6; }}
    .small {{ grid-column: span 2; }}
    .checks {{
      grid-column: span 4;
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      padding-top: 28px;
    }}
    .fieldset {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      background: rgba(255,255,255,0.65);
      min-height: 46px;
    }}
    .fieldset label {{
      margin-right: 12px;
      white-space: nowrap;
    }}
    button {{
      min-height: 48px;
      border: 0;
      background: linear-gradient(135deg, var(--accent), #09584e);
      color: white;
      padding: 0 18px;
      font: inherit;
      cursor: pointer;
    }}
    .button-link {{
      min-height: 48px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      border: 0;
      background: linear-gradient(135deg, var(--accent), #09584e);
      color: white;
      padding: 0 18px;
      font: inherit;
    }}
    .button-link-secondary {{
      background: linear-gradient(135deg, #9b5a23, #7f4312);
    }}
    .results {{
      margin-top: 20px;
      display: grid;
      gap: 18px;
    }}
    .band {{
      padding: 18px;
      border-radius: 8px;
    }}
    .stats {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }}
    .stat {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
      padding: 14px;
      display: grid;
      gap: 4px;
      min-height: 86px;
    }}
    .stat strong {{
      font-size: 1.7rem;
      line-height: 1;
    }}
    .stat span {{
      color: var(--muted);
    }}
    .section-head {{
      display: flex;
      gap: 16px;
      justify-content: space-between;
      align-items: start;
      flex-wrap: wrap;
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .chip {{
      border: 1px solid color-mix(in srgb, var(--accent) 45%, white 55%);
      padding: 6px 10px;
      font-size: 0.9rem;
      background: rgba(13, 107, 95, 0.08);
    }}
    .compact {{
      margin: 10px 0 6px;
    }}
    .badge {{
      border: 1px solid color-mix(in srgb, var(--accent) 45%, white 55%);
      padding: 4px 8px;
      font-size: 0.78rem;
      background: rgba(13, 107, 95, 0.08);
    }}
    .badge-muted {{
      border-color: color-mix(in srgb, var(--muted) 35%, white 65%);
      background: rgba(93, 102, 95, 0.08);
    }}
    .badge-warn {{
      border-color: color-mix(in srgb, var(--accent-2) 40%, white 60%);
      background: rgba(199, 93, 44, 0.12);
    }}
    .bullets, .warnings {{
      padding-left: 18px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: 1.4fr 0.8fr;
    }}
    .downloads {{
      display: grid;
      gap: 14px;
    }}
    .download-actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .evidence {{
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }}
    .meta, .muted, .placeholder {{
      color: var(--muted);
    }}
    .papers {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    .paper {{
      border: 1px solid var(--line);
      padding: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .paper-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--accent-2);
      font-size: 0.88rem;
    }}
    .links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
    }}
    details {{
      margin-top: 18px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      padding: 12px;
    }}
    pre {{
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.85rem;
      line-height: 1.45;
    }}
    .error {{
      color: #8b1e1e;
      background: #fff0ee;
      border: 1px solid #e2b9b2;
      padding: 14px;
    }}
    @media (max-width: 900px) {{
      .query, .themes, .small, .checks {{ grid-column: span 12; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>Biomedical Research Navigator</h1>
      <p class="lede">Search and organize biomedical literature from trusted public sources, review evidence-rich results, and export structured findings for academic and clinical research workflows.</p>
      <form method="get" action="/">
        <div class="row">
          <div class="field query">
            <label for="query">Query</label>
            <input id="query" type="text" name="query" value="{html.escape(query, quote=True)}" placeholder="liver cancer immunotherapy">
          </div>
          <div class="field themes">
            <label for="theme_terms">Theme terms (comma-separated)</label>
            <input id="theme_terms" type="text" name="theme_terms" value="{html.escape(theme_terms, quote=True)}" placeholder="immunotherapy, biomarkers, hepatocellular carcinoma">
          </div>
        </div>
        <div class="row">
          <div class="field small">
            <label for="max_results">Max results</label>
            <input id="max_results" type="number" min="1" max="100" name="max_results" value="{max_results}">
          </div>
          <div class="field small">
            <label for="year_from">Year from</label>
            <input id="year_from" type="number" min="1800" max="2100" name="year_from" value="{year_from or ''}">
          </div>
          <div class="field small">
            <label for="year_to">Year to</label>
            <input id="year_to" type="number" min="1800" max="2100" name="year_to" value="{year_to or ''}">
          </div>
          <div class="checks">
            <label><input type="checkbox" name="open_access_only" {checked}> Open access only</label>
            <label><input type="checkbox" name="fetch_full_text" {checked_ft}> Fetch free full text</label>
          </div>
          <fieldset class="fieldset small">
            <legend>Sources</legend>
            <label><input type="checkbox" name="sources" value="europe_pmc" {europe_checked}> Europe PMC</label>
            <label><input type="checkbox" name="sources" value="pubmed" {pubmed_checked}> PubMed</label>
            <label><input type="checkbox" name="sources" value="google_scholar" {scholar_checked}> Scholar link</label>
          </fieldset>
          <div class="checks">
            <button type="submit">Run search</button>
          </div>
        </div>
      </form>
    </section>
    <section class="results">
      {rendered_results}
      <details>
        <summary>Raw profile JSON</summary>
        <pre>{html.escape(profile_json)}</pre>
      </details>
    </section>
  </main>
</body>
</html>
"""


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_int(value: str) -> int | None:
    try:
        return int(value) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    logger.info("Starting preview server at http://%s:%s", HOST, PORT)
    server = ThreadingHTTPServer((HOST, PORT), PreviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down preview server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
