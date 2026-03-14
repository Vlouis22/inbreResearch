"""
Medical T5 summarization model wrapper.

Model: Falconsai/medical_summarization

This model's HuggingFace config does not declare a pipeline_tag that
newer versions of `transformers` recognise as "summarization", so using
`pipeline("summarization", model=...)` raises an "Unknown task" error.

loads the model directly via AutoModelForSeq2SeqLM + AutoTokenizer and
call model.generate() explicitly.  This bypasses the pipeline task registry
entirely and works with any version of transformers.
"""

import logging

from config import SUMMARIZATION_MAX_INPUT_TOKENS, SUMMARIZATION_MAX_LENGTH, SUMMARIZATION_MIN_LENGTH

logger = logging.getLogger(__name__)

# Lazy-loaded model and tokenizer references
_model = None
_tokenizer = None


def load_medical_t5(model_name: str = "Falconsai/medical_summarization") -> None:
    """
    Load the Medical T5 model and tokenizer directly.
    Safe to call multiple times; only loads once.
    """
    global _model, _tokenizer
    if _model is not None:
        return

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        logger.info("Loading Medical T5 model: %s", model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _model.eval()
        logger.info("Medical T5 loaded successfully.")
    except Exception as exc:
        logger.warning(
            "Could not load Medical T5 (%s). Summarization will use extractive fallback. "
            "Error: %s",
            model_name,
            exc,
        )
        _model = None
        _tokenizer = None


def summarize(text: str) -> str:
    """
    Generate a medical summary of *text*.

    Text is truncated to the model's token budget before encoding.
    Falls back to extractive summarization if the model is unavailable.
    """
    if not text.strip():
        return "No clinical text provided for summarization."

    truncated = _truncate_words(text, max_words=SUMMARIZATION_MAX_INPUT_TOKENS)

    if _model is not None and _tokenizer is not None:
        try:
            import torch

            inputs = _tokenizer(
                truncated,
                return_tensors="pt",
                max_length=SUMMARIZATION_MAX_INPUT_TOKENS,
                truncation=True,
            )

            with torch.no_grad():
                output_ids = _model.generate(
                    inputs["input_ids"],
                    max_length=SUMMARIZATION_MAX_LENGTH,
                    min_length=SUMMARIZATION_MIN_LENGTH,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            summary: str = _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            logger.info("Medical T5 summarization complete (%d chars).", len(summary))
            return summary

        except Exception as exc:
            logger.error("Medical T5 inference error: %s", exc)

    logger.warning("Medical T5 unavailable — using extractive fallback.")
    return _extractive_fallback(truncated)


def _truncate_words(text: str, max_words: int) -> str:
    """Truncate text to at most *max_words* words."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def _extractive_fallback(text: str) -> str:
    """
    Simple extractive summarization: return the first 3 sentences.
    Used when the T5 model cannot be loaded.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:3])
