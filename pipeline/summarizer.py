"""
Medical summarizer — Pipeline 2.

Improvements over the previous version:
  • Deduplicates source sentences before combining so the T5 model
    receives clean, non-repetitive input.
  • Post-processes the T5 output to remove near-duplicate sentences
    that the model sometimes generates when input text is repetitive.
  • Falls back to a clean extractive summary if T5 is unavailable.
"""

import logging
import re

from models.medical_t5_model import summarize
from utils.text_utils import join_texts

logger = logging.getLogger(__name__)


def build_combined_text(normalized_texts: list[str]) -> str:
    """
    Combine all normalized clinical texts into one patient document.

    Sentences that are exact duplicates across sources are removed before
    joining so the T5 model receives non-repetitive input.

    Args:
        normalized_texts: One string per input source.

    Returns:
        A single combined string, deduplicated at the sentence level.
    """
    # Filter empty strings
    non_empty = [t.strip() for t in (normalized_texts or []) if t and t.strip()]
    if not non_empty:
        return ""

    # Collect all sentences across sources, deduplicating exactly-repeated ones
    seen_sentences: set[str] = set()
    deduplicated_sources: list[str] = []

    for source_text in non_empty:
        sentences = _split_sentences(source_text)
        unique_sentences = []
        for s in sentences:
            key = s.strip().lower()
            if key and key not in seen_sentences:
                seen_sentences.add(key)
                unique_sentences.append(s.strip())
        if unique_sentences:
            deduplicated_sources.append(" ".join(unique_sentences))

    combined = join_texts(deduplicated_sources, separator="\n\n")
    logger.info("Combined text: %d characters from %d source(s).",
                len(combined), len(deduplicated_sources))
    return combined


def generate_summary(normalized_texts: list[str]) -> str:
    """
    Build the combined patient text and generate a medical summary via T5.

    Post-processes the output to remove duplicate or near-duplicate sentences.

    Args:
        normalized_texts: Normalized clinical texts from all input sources.

    Returns:
        A concise, coherent patient summary string.
    """
    combined_text = build_combined_text(normalized_texts)
    if not combined_text.strip():
        return "No clinical text available for summarization."

    logger.info("Generating medical summary via Medical T5...")
    raw_summary = summarize(combined_text)

    # Post-process: remove duplicate sentences the model may have generated
    clean_summary = _deduplicate_summary(raw_summary)

    logger.info("Medical summary ready (%d characters).", len(clean_summary))
    return clean_summary


# ── sentence-level helpers ────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences on .!? boundaries."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _deduplicate_summary(summary: str) -> str:
    """
    Remove duplicate and near-duplicate sentences from *summary*.

    Two sentences are considered near-duplicates if one contains more than
    80 % of the words of the other (simple token-overlap check).
    The first occurrence (usually the cleaner T5 generation) is kept.
    """
    if not summary or not summary.strip():
        return summary

    sentences = _split_sentences(summary)
    if not sentences:
        return summary

    kept: list[str] = []
    kept_tokens: list[set[str]] = []

    for sentence in sentences:
        tokens = set(re.findall(r"\b\w+\b", sentence.lower()))
        if not tokens:
            continue

        is_duplicate = False
        for existing_tokens in kept_tokens:
            if not existing_tokens:
                continue
            overlap = len(tokens & existing_tokens)
            # Near-duplicate: >80% token overlap in either direction
            if (overlap / len(tokens) > 0.80 or
                    overlap / len(existing_tokens) > 0.80):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(sentence)
            kept_tokens.append(tokens)

    clean = " ".join(kept).strip()
    # Capitalise the first character if it was lowercased by T5
    if clean and clean[0].islower():
        clean = clean[0].upper() + clean[1:]
    return clean