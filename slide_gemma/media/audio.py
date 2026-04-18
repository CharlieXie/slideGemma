"""Audio extraction (ffmpeg) and transcription (Whisper).

Both whisper libraries are soft dependencies -- import errors surface gracefully.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str) -> None:
    """Extract audio from *video_path* as 16 kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info("Audio saved to %s (%.1f MB)",
                output_path, os.path.getsize(output_path) / 1e6)


def transcribe(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
) -> list[dict]:
    """Transcribe *audio_path* returning ``[{"start", "end", "text"}, ...]``.

    Tries ``faster-whisper`` first, falls back to ``openai-whisper``.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        logger.info("Using faster-whisper (model=%s)", model_size)
        fw_model = WhisperModel(model_size, device="auto", compute_type="float16")
        segs, _info = fw_model.transcribe(audio_path, language=language)
        result = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]
        logger.info("Transcribed %d segments with faster-whisper", len(result))
        return result
    except ImportError:
        pass

    try:
        import whisper  # type: ignore[import-untyped]

        logger.info("Using openai-whisper (model=%s)", model_size)
        w_model = whisper.load_model(model_size)
        raw = w_model.transcribe(audio_path, language=language)
        result = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in raw["segments"]
        ]
        logger.info("Transcribed %d segments with openai-whisper", len(result))
        return result
    except ImportError:
        pass

    raise ImportError(
        "Audio transcription requires 'faster-whisper' or 'openai-whisper'. "
        "Install with:  pip install faster-whisper"
    )


def get_transcript_for_range(
    transcription: list[dict],
    start_time: float,
    end_time: float,
) -> str:
    """Return concatenated transcript text overlapping [start, end]."""
    parts = [
        seg["text"]
        for seg in transcription
        if seg["end"] > start_time and seg["start"] < end_time
    ]
    return " ".join(parts)
