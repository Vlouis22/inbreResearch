"""
Low-level text utility helpers shared across the pipeline.
"""

import re
from pathlib import Path


def is_audio_file(source: str) -> bool:
    """
    Return True when *source* refers to an audio file path.

    Recognised extensions: .wav, .mp3, .flac, .ogg, .m4a, .aac
    """
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    return Path(source).suffix.lower() in audio_extensions


def truncate_to_word_limit(text: str, max_words: int = 400) -> str:
    """Truncate *text* to at most *max_words* words to stay within model input limits."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def join_texts(texts: list[str], separator: str = "\n\n") -> str:
    """Concatenate a list of non-empty strings with *separator*."""
    return separator.join(t.strip() for t in texts if t.strip())


def extract_entity_text(token_labels: list[dict]) -> list[str]:
    """
    Reconstruct entity strings from a list of token-label dicts produced by a
    HuggingFace NER pipeline.  Handles B-/I- BIO prefixes.

    Each dict is expected to have at minimum:
        {"word": str, "entity": str, "score": float}
    """
    entities: list[str] = []
    current_tokens: list[str] = []
    current_label: str = ""

    for item in token_labels:
        word: str = item.get("word", "")
        label: str = item.get("entity", "")

        # Strip BIO prefix
        bio_prefix = label[:2] if len(label) > 2 and label[1] == "-" else ""
        base_label = label[2:] if bio_prefix else label

        if bio_prefix == "B-" or (not bio_prefix and base_label != current_label):
            # Save previous entity
            if current_tokens:
                entities.append(_join_wordpiece_tokens(current_tokens))
            current_tokens = [word]
            current_label = base_label
        elif bio_prefix == "I-" or (bio_prefix == "" and base_label == current_label):
            current_tokens.append(word)
        else:
            if current_tokens:
                entities.append(_join_wordpiece_tokens(current_tokens))
            current_tokens = []
            current_label = ""

    if current_tokens:
        entities.append(_join_wordpiece_tokens(current_tokens))

    # Filter out single-character noise and empty strings
    return [e for e in entities if len(e) > 1]


def _join_wordpiece_tokens(tokens: list[str]) -> str:
    """
    Merge WordPiece sub-tokens (those starting with ##) into a single word,
    then clean up spacing around punctuation.
    """
    merged = ""
    for token in tokens:
        if token.startswith("##"):
            merged += token[2:]
        else:
            merged += " " + token
    merged = merged.strip()
    # Remove spaces before punctuation
    merged = re.sub(r"\s([?.!,;:])", r"\1", merged)
    return merged
