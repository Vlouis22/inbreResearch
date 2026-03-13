"""
Text normalizer — cleans and standardizes raw clinical text before it
enters the model pipelines.

Responsibilities:
  • Remove non-printable / control characters
  • Normalize whitespace (tabs, multiple spaces, newlines)
  • Strip leading / trailing whitespace
  • Remove common ASR transcription artifacts (e.g. filler words, timestamps)
  • Lowercase abbreviation expansion (optional, extensible)
"""

import re
import logging

logger = logging.getLogger(__name__)

# Common ASR artifact patterns
_ASR_ARTIFACTS = re.compile(
    r"\[.*?\]"          # e.g. [inaudible], [ASR PLACEHOLDER …]
    r"|\buh+\b"         # filler: uh, uhh
    r"|\bum+\b"         # filler: um, umm
    r"|\bhmm+\b",       # filler: hmm
    re.IGNORECASE,
)

# Collapse any run of whitespace (spaces, tabs, newlines) to a single space
_WHITESPACE = re.compile(r"\s+")

# Non-printable control characters (but keep newlines for multi-line input)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize(text: str) -> str:
    """
    Apply all normalization steps to *text* and return the cleaned string.

    Steps applied in order:
      1. Strip control characters
      2. Remove ASR transcription artifacts
      3. Collapse whitespace
      4. Strip outer whitespace
    """
    if not text:
        return ""

    # 1. Remove control characters
    text = _CONTROL_CHARS.sub(" ", text)

    # 2. Remove ASR artifacts
    text = _ASR_ARTIFACTS.sub(" ", text)

    # 3. Collapse multiple whitespace characters
    text = _WHITESPACE.sub(" ", text)

    # 4. Final strip
    text = text.strip()

    logger.debug("Normalized text length: %d chars", len(text))
    return text


def normalize_all(texts: list[str]) -> list[str]:
    """Apply :func:`normalize` to every string in *texts*."""
    return [normalize(t) for t in texts]
