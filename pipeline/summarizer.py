"""
Medical summarizer — Pipeline 2.

Combines all normalized clinical text from every input source into a
single document and generates a concise patient summary using Medical T5.
"""

import logging

from models.medical_t5_model import summarize
from utils.text_utils import join_texts

logger = logging.getLogger(__name__)


def build_combined_text(normalized_texts: list[str]) -> str:
    """
    Concatenate all normalized clinical texts into one combined patient document.

    Args:
        normalized_texts: One normalized string per input source, in the order
            [doctor_notes, patient_conversation, health_records].

    Returns:
        A single string containing all clinical text separated by blank lines.
    """
    combined = join_texts(normalized_texts, separator="\n\n")
    logger.info("Combined text length: %d characters.", len(combined))
    return combined


def generate_summary(normalized_texts: list[str]) -> str:
    """
    Combine all normalized texts and generate a medical summary.

    Args:
        normalized_texts: Normalized clinical texts from all input sources.

    Returns:
        A free-text patient summary produced by Medical T5
        (or the extractive fallback if the model is unavailable).
    """
    combined_text = build_combined_text(normalized_texts)
    if not combined_text.strip():
        return "No clinical text available for summarization."

    logger.info("Generating medical summary via Medical T5...")
    summary = summarize(combined_text)
    logger.info("Medical summary generated (%d characters).", len(summary))
    return summary
