"""Unified transcription utilities - both live and batch."""

import time
from pathlib import Path
from typing import Any

from .audio import SAMPLE_RATE, detect_speech_segments, filter_segments_by_speech


def transcribe_audio(
    audio_path: Path,
    model: str = "mlx-community/distil-whisper-large-v3",
    language: str = "en",
    detect_speech: bool = False,
) -> dict[str, Any]:
    """Transcribe audio file with MLX Whisper.

    Args:
        audio_path: Path to audio file
        model: Whisper model to use
        language: Language code (en, pt, es, etc.)
        detect_speech: If True, use VAD to filter out silence (for mic audio)

    Returns:
        dict with 'text', 'segments', 'duration_seconds', 'language'
    """
    import mlx_whisper

    # Speech detection for mic audio
    speech_ranges = None
    if detect_speech:
        speech_ranges = detect_speech_segments(audio_path)
        if not speech_ranges:
            return {"text": "", "segments": [], "duration_seconds": 0, "language": language}

    start = time.time()
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language=language,
        task="transcribe",
        verbose=False,
    )
    duration = time.time() - start

    # Filter segments by speech if VAD was used
    segments = result.get("segments", [])
    if speech_ranges:
        segments = filter_segments_by_speech(segments, speech_ranges)

    return {
        "text": result["text"].strip(),
        "segments": segments,
        "duration_seconds": duration,
        "language": result.get("language", language),
    }


def format_transcript(segments: list[dict]) -> str:
    """Format segments with timestamps: [MM:SS] text"""
    lines = []
    for seg in segments:
        seconds = int(seg["start"])
        mm, ss = divmod(seconds, 60)
        text = seg["text"].strip()
        if text:
            lines.append(f"[{mm:02d}:{ss:02d}] {text}")
    return "\n".join(lines)


def get_best_model_for_language(lang: str, default_model: str) -> str:
    """Get the best Whisper model for a language.

    Distil-whisper is English-only, so switch to turbo for other languages.
    """
    if lang != "en" and "distil" in default_model:
        return default_model.replace("distil-whisper-large-v3", "whisper-large-v3-turbo")
    return default_model
