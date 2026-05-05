"""
Deterministic extractive synthesis for literature results.

This deliberately avoids paid LLM APIs and local model requirements. It ranks
sentences from abstracts/free full text by query and theme-term evidence, then
builds cited bullets from those snippets.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from schemas.literature_profile import (
    EvidenceSnippet,
    LiteratureSearchRequest,
    LiteratureSynthesis,
    PaperMetadata,
)

_SECTION_BOOSTS = {
    "results": 2.0,
    "result": 2.0,
    "findings": 2.0,
    "discussion": 1.5,
    "conclusion": 1.5,
    "conclusions": 1.5,
    "abstract": 1.0,
}

_STOP_WORDS = {
    "about", "after", "against", "among", "available", "because", "between",
    "cancer", "could", "during", "from", "have", "into", "liver", "more",
    "paper", "papers", "patient", "patients", "research", "result", "results",
    "study", "studies", "that", "their", "there", "these", "this", "through",
    "treatment", "using", "were", "what", "when", "where", "which", "with",
}


def synthesize_literature(
    request: LiteratureSearchRequest,
    papers: list[PaperMetadata],
    max_snippets: int = 12,
) -> tuple[list[EvidenceSnippet], LiteratureSynthesis, list[str]]:
    """Return ranked evidence snippets and an evidence-backed synthesis."""
    warnings: list[str] = []
    terms = _build_terms(request)
    snippets = _rank_snippets(papers, terms)
    selected = _select_snippets(snippets, max_snippets=max_snippets)

    if not papers:
        limitations = ["No papers were retrieved from the enabled sources."]
    else:
        limitations = [
            "Synthesis is extractive and limited to abstracts plus legally available free full text.",
            "Paywalled full text is not downloaded or inferred.",
        ]

    if not selected:
        synthesis = LiteratureSynthesis(
            query=request.query,
            top_themes=[],
            bullets=[],
            limitations=limitations + ["No evidence sentence matched the query/theme terms strongly enough."],
            paper_table=_build_paper_table(papers),
        )
        return [], synthesis, warnings

    top_themes = _rank_themes(selected)
    bullets = _build_bullets(top_themes, selected)
    synthesis = LiteratureSynthesis(
        query=request.query,
        top_themes=top_themes[:8],
        bullets=bullets,
        limitations=limitations,
        paper_table=_build_paper_table(papers),
    )
    return selected, synthesis, warnings


def _build_terms(request: LiteratureSearchRequest) -> list[str]:
    terms: list[str] = []
    for theme in request.theme_terms:
        theme = theme.strip().lower()
        if theme:
            terms.append(theme)

    query_words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", request.query.lower())
    for word in query_words:
        if len(word) >= 4 and word not in _STOP_WORDS:
            terms.append(word)

    return _dedup(terms)


def _rank_snippets(papers: list[PaperMetadata], terms: list[str]) -> list[EvidenceSnippet]:
    ranked: list[EvidenceSnippet] = []
    seen_sentences: set[str] = set()
    per_paper_counts: Counter[str] = Counter()

    for paper in papers:
        paper_terms = " ".join([paper.title, " ".join(paper.mesh_terms), " ".join(paper.keywords)]).lower()
        for section, sentence in _iter_candidate_sentences(paper):
            normalized = _normalize_sentence_key(sentence)
            if normalized in seen_sentences:
                continue
            matched = _matched_terms(sentence, terms)
            if not matched:
                continue
            score = float(len(matched) * 2)
            score += sum(1.0 for term in matched if term in paper_terms)
            score += _section_score(section)
            score += min(len(sentence) / 240.0, 1.0)
            per_paper_counts[paper.paper_id] += 1
            snippet = EvidenceSnippet(
                snippet_id="",
                paper_id=paper.paper_id,
                section=section,
                text=sentence,
                score=round(score, 3),
                matched_terms=matched,
            )
            ranked.append(snippet)
            seen_sentences.add(normalized)

    ranked.sort(key=lambda item: (-item.score, item.paper_id, item.text))
    for index, snippet in enumerate(ranked, start=1):
        snippet.snippet_id = f"E{index}"
    return ranked


def _select_snippets(snippets: list[EvidenceSnippet], max_snippets: int) -> list[EvidenceSnippet]:
    selected: list[EvidenceSnippet] = []
    per_paper: Counter[str] = Counter()
    for snippet in snippets:
        if len(selected) >= max_snippets:
            break
        if per_paper[snippet.paper_id] >= 3:
            continue
        selected.append(snippet)
        per_paper[snippet.paper_id] += 1
    for index, snippet in enumerate(selected, start=1):
        snippet.snippet_id = f"E{index}"
    return selected


def _iter_candidate_sentences(paper: PaperMetadata) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for sentence in _split_sentences(paper.abstract):
        candidates.append(("abstract", sentence))

    current_section = "full text"
    for raw_line in (paper.full_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\[(?P<section>[^\]]+)\]\s*(?P<text>.*)$", line)
        if match:
            current_section = match.group("section").strip().lower()
            line = match.group("text").strip()
        for sentence in _split_sentences(line):
            candidates.append((current_section, sentence))
    return candidates


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if 35 <= len(sentence.strip()) <= 500]


def _matched_terms(sentence: str, terms: list[str]) -> list[str]:
    lower = sentence.lower()
    matched = []
    for term in terms:
        if " " in term:
            if term in lower:
                matched.append(term)
            continue
        if re.search(rf"\b{re.escape(term)}\b", lower):
            matched.append(term)
    return matched


def _section_score(section: str) -> float:
    lower = section.lower()
    for key, boost in _SECTION_BOOSTS.items():
        if key in lower:
            return boost
    return 0.0


def _rank_themes(snippets: list[EvidenceSnippet]) -> list[str]:
    paper_sets: dict[str, set[str]] = defaultdict(set)
    scores: Counter[str] = Counter()
    for snippet in snippets:
        for term in snippet.matched_terms:
            paper_sets[term].add(snippet.paper_id)
            scores[term] += int(snippet.score * 10)
    return sorted(paper_sets, key=lambda term: (-len(paper_sets[term]), -scores[term], term))


def _build_bullets(top_themes: list[str], snippets: list[EvidenceSnippet]) -> list[str]:
    bullets: list[str] = []
    for theme in top_themes[:5]:
        theme_snippets = [snippet for snippet in snippets if theme in snippet.matched_terms]
        paper_ids = sorted({snippet.paper_id for snippet in theme_snippets})
        evidence_ids = ", ".join(snippet.snippet_id for snippet in theme_snippets[:3])
        bullets.append(
            f"{theme}: supported by {len(paper_ids)} paper(s) "
            f"({', '.join(paper_ids[:5])}); see evidence {evidence_ids}."
        )
    return bullets


def _build_paper_table(papers: list[PaperMetadata]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for paper in papers:
        rows.append(
            {
                "paper_id": paper.paper_id,
                "year": str(paper.year or ""),
                "title": paper.title,
                "journal": paper.journal,
                "doi": paper.doi or "",
                "pmid": paper.pmid or "",
                "pmcid": paper.pmcid or "",
                "source": paper.source,
                "free_full_text": "yes" if paper.has_free_full_text else "no",
            }
        )
    return rows


def _normalize_sentence_key(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out
