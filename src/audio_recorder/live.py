"""Live transcription using Moonshine (en) or Whisper (multilingual)."""

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .audio import SAMPLE_RATE

OVERLAP_SECONDS = 1  # Context for chunk boundaries


@dataclass
class LiveTranscriber:
    """Real-time transcription that auto-selects engine based on language."""

    language: str = "en"
    update_interval: float = 3.0
    output_path: Path | None = None

    # Internal
    _model: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)
    _chunks: list[str] = field(default_factory=list)
    _last_processed: int = 0
    _last_words: list[str] = field(default_factory=list)
    _stats: dict = field(default_factory=lambda: {"time": 0.0, "count": 0})
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _worker: threading.Thread | None = field(default=None, repr=False)
    _result: tuple[str | None, float] | None = None

    def load_model(self) -> float:
        """Load transcription model."""
        start = time.time()

        if self.language == "en":
            # Use Moonshine for English (faster, better quality)
            from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

            self._model = MoonshineOnnxModel(model_name="moonshine/base")
            self._tokenizer = load_tokenizer()
        else:
            # Use Whisper-tiny for other languages (24x realtime)
            import mlx_whisper  # Model loads on first use
            self._model = "mlx-community/whisper-tiny-mlx"

        return time.time() - start

    def process_buffer(self, get_range_np_fn, buffer_length: int) -> tuple[str | None, float]:
        """Process new audio chunk (async)."""
        result = self._collect_result()

        if self._worker and self._worker.is_alive():
            return result or (None, 0)

        if buffer_length < SAMPLE_RATE:
            return result or (None, 0)

        # Get chunk with overlap for context
        overlap_samples = int(OVERLAP_SECONDS * SAMPLE_RATE)
        chunk_start = max(self._last_processed - overlap_samples, 0)
        has_overlap = self._last_processed > 0

        samples = get_range_np_fn(chunk_start, buffer_length)
        if len(samples) < SAMPLE_RATE:
            return result or (None, 0)

        # Launch background inference
        self._worker = threading.Thread(
            target=self._infer,
            args=(samples, buffer_length, has_overlap),
            daemon=True,
        )
        self._worker.start()
        return result or (None, 0)

    def process_buffer_sync(self, get_range_np_fn, buffer_length: int) -> tuple[str | None, float]:
        """Process final chunk (sync)."""
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=10)
        result = self._collect_result()

        if buffer_length >= SAMPLE_RATE:
            overlap_samples = int(OVERLAP_SECONDS * SAMPLE_RATE)
            chunk_start = max(self._last_processed - overlap_samples, 0)
            has_overlap = self._last_processed > 0
            samples = get_range_np_fn(chunk_start, buffer_length)

            if len(samples) >= SAMPLE_RATE:
                self._infer(samples, buffer_length, has_overlap)
                final = self._collect_result()
                if result and final:
                    return (f"{result[0]} {final[0]}", result[1] + final[1])
                return final or result or (None, 0)

        return result or (None, 0)

    def _infer(self, audio, buffer_length: int, has_overlap: bool):
        """Run inference (Moonshine or Whisper)."""
        start = time.time()
        try:
            if self.language == "en":
                text = self._infer_moonshine(audio)
            else:
                text = self._infer_whisper(audio)

            elapsed = time.time() - start
            self._stats["time"] += elapsed
            self._stats["count"] += 1
            self._last_processed = buffer_length

            if not text:
                with self._lock:
                    self._result = (None, elapsed)
                return

            # Strip overlap, repetitions, normalize
            if has_overlap and self._last_words:
                text = self._strip_overlap(text)
            text = self._strip_repetitions(text)
            text = self._normalize_boundary(text)

            if text:
                self._last_words = text.split()[-10:]
                self._chunks.append(text)
                self._write()

            with self._lock:
                self._result = (text, elapsed)

        except Exception as e:
            with self._lock:
                self._result = (None, 0)
            print(f"  > transcription error: {e}")

    def _infer_moonshine(self, audio) -> str:
        """Moonshine inference (English only)."""
        max_len = int(len(audio) / SAMPLE_RATE * 5) + 10
        audio = audio.reshape(1, -1)
        tokens = self._model.generate(audio, max_len=max_len)
        return self._tokenizer.decode_batch(tokens)[0].strip()

    def _infer_whisper(self, audio) -> str:
        """Whisper inference (multilingual)."""
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model,
            language=self.language,
            task="transcribe",
            verbose=False,
        )
        return result["text"].strip()

    def _strip_overlap(self, text: str) -> str:
        """Remove overlapping words from previous chunk."""
        words = text.split()
        if not words:
            return text

        # Try exact word match
        prev_norm = [w.lower().strip(".,!?;:") for w in self._last_words]
        new_norm = [w.lower().strip(".,!?;:") for w in words]

        best = 0
        for length in range(1, min(len(prev_norm), len(new_norm)) + 1):
            if prev_norm[-length:] == new_norm[:length]:
                best = length

        if best > 0:
            return " ".join(words[best:])

        # Try substring match (3+ words)
        for i in range(min(10, len(new_norm))):
            match_len = 0
            for j in range(min(len(prev_norm), len(new_norm) - i)):
                if prev_norm[j] == new_norm[i + j]:
                    match_len += 1
                else:
                    break
            if match_len >= 3:
                return " ".join(words[i + match_len :])

        return text

    @staticmethod
    def _strip_repetitions(text: str) -> str:
        """Remove hallucinated repetition loops."""
        for n in (3, 2, 1):
            min_reps = 1 if n == 1 else 2
            pattern = r"((?:\S+\s+){" + str(n - 1) + r"}\S+)(\s+\1){" + str(min_reps) + r",}"
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)
        return text.strip()

    def _normalize_boundary(self, text: str) -> str:
        """Fix casing and punctuation at chunk boundaries."""
        if not text:
            return text

        # Remove trailing period artifact
        if text.endswith(".") and not text.endswith("..."):
            text = text[:-1].rstrip()

        # Lowercase start if continuing sentence
        if self._chunks and text:
            prev = self._chunks[-1].rstrip()
            if prev and prev[-1] not in ".!?":
                text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()

        return text

    def _collect_result(self):
        """Get pending result."""
        with self._lock:
            result = self._result
            self._result = None
        return result

    def _write(self):
        """Write transcript to file."""
        if self.output_path:
            self.output_path.write_text(" ".join(self._chunks))

    def get_transcript(self) -> str:
        """Get complete transcript."""
        return " ".join(self._chunks)

    def get_stats(self) -> dict:
        """Get performance stats."""
        t, n = self._stats["time"], self._stats["count"]
        return {
            "updates": n,
            "avg_time": t / n if n else 0,
            "total_time": t,
        }
