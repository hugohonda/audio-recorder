"""Live transcription using Moonshine with overlapping chunks."""

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .audio import SAMPLE_RATE

# Seconds of already-transcribed audio to include as context in each chunk.
# Gives Moonshine sentence context so words at boundaries don't get cut.
OVERLAP_SECONDS = 1


def _normalize_word(w: str) -> str:
    """Strip punctuation and lowercase for comparison."""
    return re.sub(r"[^\w]", "", w.lower())


@dataclass
class MoonshineTranscriber:
    """
    Live transcription using Moonshine with overlapping chunks.

    Each update transcribes (overlap + new) audio, then strips the
    overlapping words via suffix matching. This keeps chunks connected
    while processing a fixed amount of audio per update.
    """

    model_name: str = "moonshine/base"
    update_interval: float = 3.0  # seconds between transcription updates
    output_path: Path | None = None

    # Internal state
    _model: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)
    _chunks: list[str] = field(default_factory=list)
    _last_processed: int = 0  # sample index up to which we've transcribed
    _last_words: list[str] = field(default_factory=list)  # tail words for overlap matching
    _total_process_time: float = 0.0
    _update_count: int = 0
    _total_audio_seconds: float = 0.0
    _inference_lock: threading.Lock = field(default_factory=threading.Lock)
    _pending_result: tuple[str | None, float] | None = field(default=None, repr=False)
    _worker: threading.Thread | None = field(default=None, repr=False)

    def load_model(self, *, quiet: bool = False) -> float:
        """Load Moonshine model. Returns load time in seconds."""
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

        if not quiet:
            print(f"Loading Moonshine ({self.model_name})...")

        start = time.time()
        self._model = MoonshineOnnxModel(model_name=self.model_name)
        self._tokenizer = load_tokenizer()
        load_time = time.time() - start

        if not quiet:
            print(f"Moonshine ready ({load_time:.1f}s)")
        return load_time

    def process_buffer(self, get_range_np_fn, buffer_length: int) -> tuple[str | None, float]:
        """Transcribe new audio with overlap context. Returns (new_text, process_time).

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

        # Include overlap from already-processed audio for context
        overlap_samples = int(OVERLAP_SECONDS * SAMPLE_RATE)
        chunk_start = max(self._last_processed - overlap_samples, 0)
        has_overlap = self._last_processed > 0 and chunk_start < self._last_processed
        actual_overlap_seconds = (self._last_processed - chunk_start) / SAMPLE_RATE

        samples = get_range_np_fn(chunk_start, buffer_length)
        if len(samples) < SAMPLE_RATE:
            return result if result else (None, 0)

        self._total_audio_seconds = buffer_length / SAMPLE_RATE
        audio = samples.reshape(1, -1)
        audio_seconds = len(samples) / SAMPLE_RATE

        # Launch inference in background thread
        self._worker = threading.Thread(
            target=self._infer,
            args=(audio, audio_seconds, actual_overlap_seconds, has_overlap, buffer_length),
            daemon=True,
        )
        self._worker.start()

        return result if result else (None, 0)

    def process_buffer_sync(self, get_range_np_fn, buffer_length: int) -> tuple[str | None, float]:
        """Synchronous version -- used for the final flush when recording stops."""
        # Drain any background work first
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=5.0)
        result = self._collect_result()

        if buffer_length < SAMPLE_RATE:
            return result if result else (None, 0)

        overlap_samples = int(OVERLAP_SECONDS * SAMPLE_RATE)
        chunk_start = max(self._last_processed - overlap_samples, 0)
        has_overlap = self._last_processed > 0 and chunk_start < self._last_processed
        actual_overlap_seconds = (self._last_processed - chunk_start) / SAMPLE_RATE

        samples = get_range_np_fn(chunk_start, buffer_length)
        if len(samples) < SAMPLE_RATE:
            return result if result else (None, 0)

        self._total_audio_seconds = buffer_length / SAMPLE_RATE
        audio = samples.reshape(1, -1)
        audio_seconds = len(samples) / SAMPLE_RATE

        self._infer(audio, audio_seconds, actual_overlap_seconds, has_overlap, buffer_length)
        final = self._collect_result()
        # Merge: if we had a prior result and a final one, return both
        if result and final:
            r_text, f_text = result[0], final[0]
            if r_text and f_text:
                combined = r_text + " " + f_text
            else:
                combined = f_text or r_text
            return combined, result[1] + final[1]
        return final if final else (result if result else (None, 0))

    def _infer(
        self,
        audio,
        audio_seconds: float,
        overlap_seconds: float,
        has_overlap: bool,
        buffer_length: int,
    ) -> None:
        """Run model inference and post-process (called from background thread)."""
        # Cap max tokens proportional to audio length to prevent hallucination.
        max_len = int(audio_seconds * 5) + 10

        start = time.time()
        try:
            tokens = self._model.generate(audio, max_len=max_len)
            text = self._tokenizer.decode_batch(tokens)[0].strip()
            process_time = time.time() - start

            self._total_process_time += process_time
            self._update_count += 1
            self._last_processed = buffer_length

            if not text:
                with self._inference_lock:
                    self._pending_result = (None, process_time)
                return

            # Strip overlapping words from the beginning of this chunk's text
            if has_overlap and self._last_words:
                text = self._strip_overlap(text, overlap_seconds, audio_seconds)

            if not text:
                with self._inference_lock:
                    self._pending_result = (None, process_time)
                return

            # Collapse hallucinated repetition loops
            text = self._strip_repetitions(text)

            if not text:
                with self._inference_lock:
                    self._pending_result = (None, process_time)
                return

            # Normalize chunk boundary: fix casing and trailing punctuation
            text = self._normalize_boundary(text)

            # Save tail words for next overlap matching
            words = text.split()
            self._last_words = words[-10:]

            self._chunks.append(text)
            self._write_transcript()

            with self._inference_lock:
                self._pending_result = (text, process_time)

        except Exception as e:
            with self._inference_lock:
                self._pending_result = (None, 0)
            print(f"\n  > moonshine error: {e}")

    def _collect_result(self) -> tuple[str | None, float] | None:
        """Collect pending result from background inference, if any."""
        with self._inference_lock:
            result = self._pending_result
            self._pending_result = None
        return result

    def _strip_overlap(self, text: str, overlap_seconds: float, total_seconds: float) -> str:
        """Remove overlapping words from the start of new text.

        Uses two strategies:
        1. Exact word matching (punctuation-stripped) between previous tail
           and new text head.
        2. Time-based fallback: estimate overlap word count from the known
           overlap/total audio ratio when exact matching fails.
        """
        words = text.split()
        if not words:
            return text

        prev_norm = [_normalize_word(w) for w in self._last_words]
        new_norm = [_normalize_word(w) for w in words]

        # Strategy 1: exact match (ignoring punctuation/case)
        best_overlap = 0
        max_check = min(len(prev_norm), len(new_norm))

        for overlap_len in range(1, max_check + 1):
            tail = prev_norm[-overlap_len:]
            head = new_norm[:overlap_len]
            if tail == head:
                best_overlap = overlap_len

        if best_overlap > 0:
            return " ".join(words[best_overlap:])

        # Strategy 2: find any subsequence of 3+ consecutive matching words
        # from _last_words appearing in the first half of the new text
        min_match = 3
        search_limit = min(len(new_norm), max(10, len(new_norm) // 2))

        for start_pos in range(search_limit):
            match_len = 0
            for j in range(min(len(prev_norm), len(new_norm) - start_pos)):
                if prev_norm[j] == new_norm[start_pos + j]:
                    match_len += 1
                else:
                    break
            if match_len >= min_match:
                # Found a matching run starting at start_pos
                split_at = start_pos + match_len
                return " ".join(words[split_at:])

        # Strategy 3: time-based estimation as last resort
        if total_seconds > 0 and overlap_seconds > 0:
            overlap_ratio = overlap_seconds / total_seconds
            est_words = int(len(words) * overlap_ratio)
            if est_words > 0:
                return " ".join(words[est_words:])

        return text

    @staticmethod
    def _strip_repetitions(text: str) -> str:
        """Collapse hallucinated repetition loops.

        Moonshine sometimes gets stuck repeating the same word or short
        phrase ("f of f of f of ..."). Detect runs of repeated 1/2/3-grams
        and collapse them to a single occurrence, keeping the text before
        the loop started.
        """
        # 3-gram, 2-gram (need 3+ occurrences), 1-gram (need 2+ to catch stutters)
        for n in (3, 2, 1):
            min_reps = 1 if n == 1 else 2
            pattern = r"((?:\S+\s+){" + str(n - 1) + r"}\S+)(\s+\1){" + str(min_reps) + r",}"
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)
        return text.strip()

    def _normalize_boundary(self, text: str) -> str:
        """Normalize casing and punctuation at chunk boundaries.

        - Strips trailing period artifacts (Moonshine adds one per chunk).
        - Lowercases first character if continuing a sentence from previous chunk.
        """
        if not text:
            return text

        # Strip trailing period artifact (keep ... and sentence-internal punctuation)
        if text.endswith(".") and not text.endswith("..."):
            text = text[:-1].rstrip()

        # If previous chunk didn't end a sentence, lowercase the start
        if self._chunks and text:
            prev = self._chunks[-1].rstrip()
            if prev and prev[-1] not in ".!?":
                text = text[0].lower() + text[1:]

        return text

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
