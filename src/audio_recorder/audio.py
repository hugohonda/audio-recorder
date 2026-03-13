"""Audio utilities: conversion, resampling, buffer."""

import threading
import wave
from pathlib import Path
from typing import Any

import lameenc
import numpy as np

# Audio format constants - optimized for transcription
SAMPLE_RATE = 16000  # Whisper/Moonshine native rate
MIC_SAMPLE_RATE = 24000  # macOS mic default rate via ScreenCaptureKit
CHANNELS = 1
MP3_BITRATE = 64  # kbps, plenty for speech


def float32_to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert float32 samples [-1.0, 1.0] to int16 values."""
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)


def bytes_to_float32(data: bytes) -> np.ndarray:
    """Convert bytes to float32 numpy array."""
    return np.frombuffer(data, dtype=np.float32).copy()


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio using vectorized linear interpolation."""
    if len(samples) == 0 or from_rate == to_rate:
        return samples

    ratio = to_rate / from_rate
    new_len = int(len(samples) * ratio)
    indices = np.arange(new_len) / ratio
    lo = indices.astype(np.intp)
    hi = np.minimum(lo + 1, len(samples) - 1)
    frac = indices - lo
    return samples[lo] * (1 - frac) + samples[hi] * frac


def save_wav(samples: np.ndarray, path: Path) -> int:
    """Save int16 samples as WAV file. Returns file size in bytes."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())
    return path.stat().st_size


def save_mp3(samples: np.ndarray, path: Path) -> int:
    """Save int16 samples as MP3 file. Returns file size in bytes."""
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(MP3_BITRATE)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)

    pcm_data = samples.tobytes()
    mp3_data = encoder.encode(pcm_data) + encoder.flush()
    path.write_bytes(mp3_data)
    return path.stat().st_size


def detect_speech_segments(
    audio_path: Path,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 100,
) -> list[tuple[float, float]]:
    """Detect speech segments using Silero VAD.

    Uses a neural-network-based voice activity detector for accurate
    speech detection. Designed for mic recordings where most of the
    audio is silence.

    Returns list of (start_seconds, end_seconds) tuples.
    """
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    model = load_silero_vad()
    wav = read_audio(str(audio_path), sampling_rate=SAMPLE_RATE)

    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )

    return [(ts["start"], ts["end"]) for ts in timestamps]


def filter_segments_by_speech(
    segments: list[dict[str, Any]],
    speech_ranges: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Keep only Whisper segments that overlap with detected speech ranges."""
    if not speech_ranges:
        return []
    filtered = []
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        for start, end in speech_ranges:
            if start <= mid <= end:
                filtered.append(seg)
                break
    return filtered


class AudioBuffer:
    """Thread-safe buffer that collects raw audio bytes.

    Stores raw float32 bytes in a bytearray (~4 bytes/sample) instead of
    Python float objects (~36 bytes/sample), reducing memory ~9x.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = bytearray()

    def add(self, data: bytes):
        """Add raw float32 audio data to buffer."""
        with self._lock:
            self._data.extend(data)

    def get_samples(self) -> np.ndarray:
        """Get all collected samples as numpy float32 array."""
        with self._lock:
            return np.frombuffer(bytes(self._data), dtype=np.float32)

    def length(self) -> int:
        """Get current sample count."""
        with self._lock:
            return len(self._data) // 4

    def get_range_np(self, start: int, end: int) -> np.ndarray:
        """Get samples in range [start, end) as numpy float32 array."""
        with self._lock:
            chunk = bytes(self._data[start * 4 : end * 4])
            return np.frombuffer(chunk, dtype=np.float32)
