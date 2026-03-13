"""Live transcription using MLX Whisper for multilingual support."""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .audio import SAMPLE_RATE


@dataclass
class WhisperLiveTranscriber:
    """
    Live transcription using MLX Whisper with multilingual support.

    Uses whisper-tiny or whisper-small for fast real-time transcription
    in languages other than English (Moonshine is English-only).
    """

    model_name: str = "mlx-community/whisper-tiny-mlx"  # Fast enough for real-time
    language: str = "pt"  # Language code (pt, es, fr, etc.)
    update_interval: float = 3.0  # seconds between transcription updates
    output_path: Path | None = None

    # Internal state
    _model_loaded: bool = field(default=False, repr=False)
    _chunks: list[str] = field(default_factory=list)
    _last_processed: int = 0  # sample index up to which we've transcribed
    _total_process_time: float = 0.0
    _update_count: int = 0
    _total_audio_seconds: float = 0.0
    _inference_lock: threading.Lock = field(default_factory=threading.Lock)
    _pending_result: tuple[str | None, float] | None = field(default=None, repr=False)
    _worker: threading.Thread | None = field(default=None, repr=False)

    def load_model(self, *, quiet: bool = False) -> float:
        """Load MLX Whisper model. Returns load time in seconds."""
        import mlx_whisper  # noqa: F401

        if not quiet:
            print(f"Loading Whisper ({self.model_name}) for {self.language}...")

        start = time.time()
        # Model will be loaded on first inference
        self._model_loaded = True
        load_time = time.time() - start

        if not quiet:
            print(f"Whisper ready ({load_time:.1f}s)")
        return load_time

    def process_buffer(self, get_range_np_fn, buffer_length: int) -> tuple[str | None, float]:
        """Transcribe new audio. Returns (new_text, process_time).

        Runs inference in a background thread. If the previous inference is
        still running, returns the pending result (if any) without blocking.
        """
        # Check for completed background result
        result = self._collect_result()

        # Skip if previous inference still running
        if self._worker is not None and self._worker.is_alive():
            return result if result else (None, 0)

        if buffer_length < SAMPLE_RATE:
            return result if result else (None, 0)

        # Get new audio since last processed
        chunk_start = self._last_processed
        samples = get_range_np_fn(chunk_start, buffer_length)

        if len(samples) < SAMPLE_RATE:
            return result if result else (None, 0)

        self._total_audio_seconds = buffer_length / SAMPLE_RATE
        audio = samples.reshape(1, -1)

        # Launch inference in background thread
        self._worker = threading.Thread(
            target=self._infer,
            args=(audio, buffer_length),
            daemon=True,
        )
        self._worker.start()

        return result if result else (None, 0)

    def process_buffer_sync(self, get_range_np_fn, buffer_length: int) -> tuple[str | None, float]:
        """Synchronous version -- used for the final flush when recording stops."""
        # Drain any background work first
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=10.0)
        result = self._collect_result()

        if buffer_length < SAMPLE_RATE:
            return result if result else (None, 0)

        chunk_start = self._last_processed
        samples = get_range_np_fn(chunk_start, buffer_length)

        if len(samples) < SAMPLE_RATE:
            return result if result else (None, 0)

        self._total_audio_seconds = buffer_length / SAMPLE_RATE
        audio = samples.reshape(1, -1)

        self._infer(audio, buffer_length)
        final = self._collect_result()

        # Merge results
        if result and final:
            r_text, f_text = result[0], final[0]
            if r_text and f_text:
                combined = r_text + " " + f_text
            else:
                combined = f_text or r_text
            return combined, result[1] + final[1]
        return final if final else (result if result else (None, 0))

    def _infer(self, audio, buffer_length: int) -> None:
        """Run MLX Whisper inference (called from background thread)."""
        import mlx_whisper

        start = time.time()
        try:
            # Transcribe just the new chunk
            result = mlx_whisper.transcribe(
                audio[0],  # Extract 1D array from batch
                path_or_hf_repo=self.model_name,
                language=self.language,
                task="transcribe",
                verbose=False,
            )

            text = result["text"].strip()
            process_time = time.time() - start

            self._total_process_time += process_time
            self._update_count += 1
            self._last_processed = buffer_length

            if not text:
                with self._inference_lock:
                    self._pending_result = (None, process_time)
                return

            # Capitalize first letter if first chunk
            if not self._chunks and text:
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            # Otherwise lowercase if continuing sentence
            elif self._chunks and text:
                prev = self._chunks[-1].rstrip()
                if prev and prev[-1] not in ".!?":
                    text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()

            self._chunks.append(text)
            self._write_transcript()

            with self._inference_lock:
                self._pending_result = (text, process_time)

        except Exception as e:
            with self._inference_lock:
                self._pending_result = (None, 0)
            print(f"\n  > whisper error: {e}")

    def _collect_result(self) -> tuple[str | None, float] | None:
        """Collect pending result from background inference, if any."""
        with self._inference_lock:
            result = self._pending_result
            self._pending_result = None
        return result

    def _write_transcript(self) -> None:
        """Write accumulated transcript to file."""
        if not self.output_path:
            return
        self.output_path.write_text(" ".join(self._chunks))

    def get_full_transcript(self) -> str:
        """Get the complete accumulated transcript."""
        return " ".join(self._chunks)

    def get_stats(self) -> dict:
        """Get transcription statistics."""
        proc = self._total_process_time
        audio = self._total_audio_seconds
        n = self._update_count
        return {
            "updates": n,
            "total_audio_seconds": audio,
            "total_process_seconds": proc,
            "realtime_factor": audio / proc if proc else 0,
            "avg_process_time": proc / n if n else 0,
        }
