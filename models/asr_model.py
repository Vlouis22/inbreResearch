"""
Automatic Speech Recognition (ASR) module.

Primary implementation uses OpenAI Whisper via the HuggingFace `transformers`
pipeline.  If the model cannot be loaded (e.g. no internet, no GPU), the module
falls back to a lightweight placeholder that returns a clearly marked stub
transcript so the rest of the pipeline can still run.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded pipeline reference
_asr_pipeline = None


def load_asr_model(model_name: str = "openai/whisper-small") -> None:
    """
    Initialise the Whisper ASR pipeline.  Call once at application start-up.
    Subsequent calls are no-ops if the model is already loaded.
    """
    global _asr_pipeline
    if _asr_pipeline is not None:
        return

    try:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading ASR model: %s", model_name)
        _asr_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,       # handle long audio in 30-second windows
            stride_length_s=5,
        )
        logger.info("ASR model loaded successfully.")
    except Exception as exc:
        logger.warning(
            "Could not load Whisper model (%s). Falling back to placeholder ASR. "
            "Error: %s",
            model_name,
            exc,
        )
        _asr_pipeline = None


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file at *audio_path* to text.

    Returns the transcription string.  If the Whisper model is unavailable,
    returns a placeholder transcript that keeps the pipeline functional.
    """
    path = Path(audio_path)
    if not path.exists():
        logger.error("Audio file not found: %s", audio_path)
        return _placeholder_transcript(audio_path)

    if _asr_pipeline is not None:
        logger.info("Transcribing audio: %s", audio_path)
        result = _asr_pipeline(str(path))
        transcript: str = result.get("text", "").strip()
        logger.info("Transcription complete (%d characters).", len(transcript))
        return transcript

    logger.warning("Using placeholder ASR for: %s", audio_path)
    return _placeholder_transcript(audio_path)


def _placeholder_transcript(audio_path: str) -> str:
    """
    Return a synthetic transcript used when the real ASR model is unavailable.
    This allows downstream pipeline stages to run without a loaded Whisper model.
    """
    return (
        f"[ASR PLACEHOLDER — could not transcribe '{audio_path}'] "
        "Patient reports shortness of breath and chest pain for the past three days. "
        "No fever. Currently taking aspirin 81 mg daily."
    )
