"""Audio capture using macOS ScreenCaptureKit."""

import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import objc
from CoreMedia import (
    CMAudioFormatDescriptionGetStreamBasicDescription,
    CMBlockBufferCopyDataBytes,
    CMBlockBufferGetDataLength,
    CMSampleBufferGetDataBuffer,
    CMSampleBufferGetFormatDescription,
)
from Foundation import NSObject
from ScreenCaptureKit import (
    SCContentFilter,
    SCShareableContent,
    SCStream,
    SCStreamConfiguration,
    SCStreamOutputTypeAudio,
    SCStreamOutputTypeMicrophone,
)

from .audio import (
    CHANNELS,
    MIC_SAMPLE_RATE,
    SAMPLE_RATE,
    AudioBuffer,
    float32_to_int16,
    resample,
    save_mp3,
    save_wav,
)
from .transcribe import format_transcript, transcribe_audio

LINE = "─" * 50
SILENCE_TIMEOUT = 600  # Stop recording after 10 minutes of silence


def safe_execute(func, error_msg: str, critical: bool = False):
    """Execute function with error handling. Returns (success, result)."""
    try:
        return True, func()
    except Exception as e:
        print(f"  > {error_msg}: {e}")
        if critical:
            traceback.print_exc()
        return False, None


class AudioStreamOutput(NSObject):
    """Handle incoming audio from ScreenCaptureKit."""

    def initWithSystemBuffer_micBuffer_(self, system_buffer, mic_buffer):
        self = objc.super(AudioStreamOutput, self).init()
        if self is None:
            return None
        self._system_buffer = system_buffer
        self._mic_buffer = mic_buffer
        self._mic_sample_rate = None
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
        if output_type not in (SCStreamOutputTypeAudio, SCStreamOutputTypeMicrophone):
            return

        try:
            block_buffer = CMSampleBufferGetDataBuffer(sample_buffer)
            if not block_buffer:
                return

            data_length = CMBlockBufferGetDataLength(block_buffer)
            if not data_length:
                return

            status, audio_data = CMBlockBufferCopyDataBytes(block_buffer, 0, data_length, None)
            if status != 0:
                return

            if output_type == SCStreamOutputTypeAudio:
                self._system_buffer.add(bytes(audio_data))
            else:
                # Detect actual mic sample rate from first buffer
                if self._mic_sample_rate is None:
                    try:
                        fmt = CMSampleBufferGetFormatDescription(sample_buffer)
                        asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmt)
                        # PyObjC returns ASBD as tuple: (sampleRate, formatID, flags, ...)
                        self._mic_sample_rate = int(asbd[0])
                    except Exception:
                        pass
                self._mic_buffer.add(bytes(audio_data))

        except Exception as e:
            print(f"  > error: {e}")

    def stream_didStopWithError_(self, stream, error):
        if error:
            print(f"  > stream error: {error.localizedDescription()}")


class AudioRecorder:
    """Record system audio + microphone."""

    def __init__(
        self,
        output_path: str,
        include_mic: bool = True,
        mic_only: bool = False,
        live: bool = False,
        final: bool = True,
        model: str = "mlx-community/distil-whisper-large-v3",
        summarize: bool = True,
        language: str = "en",
        whisper_language: str | None = None,
    ):
        self.output_path = Path(output_path)
        self.include_mic = include_mic or mic_only  # mic_only implies mic
        self.mic_only = mic_only
        self.live = live
        self.final = final
        self.model = model
        self.summarize = summarize
        self.language = language  # For summaries (en, pt-br)
        self.whisper_language = whisper_language or ("pt" if language == "pt-br" else language)  # For transcription
        self.meeting = None  # Auto-detected meeting context

        self.stream = None
        self._handler = None
        self._system_buffer = None
        self._mic_buffer = None
        self._running = False
        self._transcriber = None
        self._vad_model = None
        self._last_speech_time = None

        # Setup live transcription
        if live:
            from .live import LiveTranscriber

            live_path = self.output_path.with_stem(f"{self.output_path.stem}_live").with_suffix(".txt")
            self._transcriber = LiveTranscriber(language=self.whisper_language, output_path=live_path)

    def start(self, duration: int | None = None):
        """Start recording."""
        self._system_buffer = AudioBuffer()
        self._mic_buffer = AudioBuffer()

        # Header
        if self.mic_only:
            mode = "mic only"
        elif self.include_mic:
            mode = "system + mic"
        else:
            mode = "system only"
        parts = [mode]
        if self._transcriber:
            engine = "moonshine" if self.whisper_language == "en" else f"whisper-small ({self.whisper_language})"
            parts.append(f"live ({engine})")
        if self.final:
            parts.append(self.model.split("/")[-1])

        # Auto-detect active meeting
        meetings_file = self.output_path.parent.parent / "meetings.json"
        if meetings_file.exists():
            try:
                from .meeting import find_active_meeting, load_meetings

                meetings = load_meetings(meetings_file)
                self.meeting = find_active_meeting(meetings)
                if self.meeting:
                    parts.append(f"meeting: {self.meeting['name']}")
            except Exception as e:
                print(f"  > warning: meeting detection failed: {e}")

        print(f"\naudio-recorder | {', '.join(parts)}")

        # Preload live model
        if self._transcriber:
            model_name = "moonshine" if self.whisper_language == "en" else f"whisper-small ({self.whisper_language})"
            sys.stdout.write(f"  > loading {model_name}... ")
            sys.stdout.flush()
            load_time = self._transcriber.load_model()
            print(f"ready ({load_time:.1f}s)")

        # Setup stream
        content = self._get_shareable_content()
        displays = content.displays()
        if not displays:
            raise RuntimeError("No displays found")

        content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(displays[0], [])

        config = SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setExcludesCurrentProcessAudio_(False)
        if self.include_mic:
            config.setCaptureMicrophone_(True)
        config.setWidth_(2)
        config.setHeight_(2)
        config.setSampleRate_(SAMPLE_RATE)
        config.setChannelCount_(CHANNELS)

        self._handler = AudioStreamOutput.alloc().initWithSystemBuffer_micBuffer_(
            self._system_buffer, self._mic_buffer
        )
        self.stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, self._handler
        )

        self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._handler, SCStreamOutputTypeAudio, None, objc.nil
        )
        if self.include_mic:
            self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._handler, SCStreamOutputTypeMicrophone, None, objc.nil
            )

        self._start_stream()
        self._running = True

        print(f"  > recording to {self.output_path.name} (ctrl+c to stop)")
        print(LINE)

        try:
            self._run_loop(duration)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _run_loop(self, duration: int | None):
        """Main recording loop."""
        start = time.monotonic()
        last_update = start
        last_silence_check = start
        interval = self._transcriber.update_interval if self._transcriber else 3.0
        silence_check_interval = 10.0  # Check for silence every 10 seconds

        while self._running:
            time.sleep(0.1)
            now = time.monotonic()

            if duration and (now - start) >= duration:
                break

            # Check for silence timeout
            if (now - last_silence_check) >= silence_check_interval:
                def check_silence():
                    if self._check_silence_timeout(now):
                        mins = SILENCE_TIMEOUT // 60
                        print(f"\n  > stopping: {mins} minute{'s' if mins != 1 else ''} of silence detected")
                        return True
                    return False

                success, should_stop = safe_execute(check_silence, "warning: silence check error")
                if should_stop:
                    break
                last_silence_check = now

            # Live transcription
            if self._transcriber and (now - last_update) >= interval:
                buf, buf_len = self._get_live_buffer()

                def update_transcription():
                    text, _ = self._transcriber.process_buffer(
                        buf.get_range_np, buf_len
                    )
                    if text:
                        print(text)

                safe_execute(update_transcription, "warning: live transcription error")
                last_update = now

        # Final live update
        if self._transcriber:
            buf, buf_len = self._get_live_buffer()

            def final_update():
                text, _ = self._transcriber.process_buffer_sync(
                    buf.get_range_np, buf_len
                )
                if text:
                    print(text)

            safe_execute(final_update, "warning: final live update error")

    def _get_live_buffer(self):
        """Get the appropriate buffer for live transcription.

        For mic-only mode, resamples mic audio to 16kHz into a temporary buffer.
        Returns (buffer, length) where buffer has a get_range_np method.
        """
        if not self.mic_only:
            return self._system_buffer, self._system_buffer.length()

        # Mic-only: resample mic audio to 16kHz for transcription
        mic_rate = (
            self._handler._mic_sample_rate
            if self._handler and self._handler._mic_sample_rate
            else MIC_SAMPLE_RATE
        )
        mic_samples = self._mic_buffer.get_samples()
        if len(mic_samples) == 0:
            return self._mic_buffer, 0
        resampled = resample(mic_samples, mic_rate, SAMPLE_RATE)

        class _ResampledView:
            """Provides get_range_np over resampled audio."""
            def __init__(self, data):
                self._data = data
            def get_range_np(self, start, end):
                return self._data[start:end]

        return _ResampledView(resampled), len(resampled)

    def _detect_speech_in_buffer(self, buffer, buffer_length: int, sample_rate: int, label: str) -> bool:
        """Detect speech in audio buffer using VAD. Returns True if speech detected."""
        if buffer_length < sample_rate:
            return False

        check_samples = min(5 * sample_rate, buffer_length)
        start_idx = buffer_length - check_samples
        recent_audio = buffer.get_range_np(start_idx, buffer_length)

        if len(recent_audio) == 0:
            return False

        try:
            from silero_vad import get_speech_timestamps

            # Resample to 16kHz if needed (for mic audio at 24kHz)
            if sample_rate != SAMPLE_RATE:
                recent_audio = resample(recent_audio, sample_rate, SAMPLE_RATE)

            # Prepare tensor
            audio_tensor = recent_audio.astype(np.float32)
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.reshape(1, -1)

            timestamps = get_speech_timestamps(
                audio_tensor,
                self._vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=0.5,
                min_speech_duration_ms=300,
                return_seconds=False,
            )
            return len(timestamps) > 0
        except Exception as e:
            print(f"  > VAD error ({label}): {e}")
            return False

    def _check_silence_timeout(self, current_time: float) -> bool:
        """Check if silence timeout exceeded. Returns True if should stop recording."""
        try:
            # Load VAD model if needed
            if self._vad_model is None:
                from silero_vad import load_silero_vad
                self._vad_model = load_silero_vad()

            # Check both streams for speech
            system_speech = self._detect_speech_in_buffer(
                self._system_buffer, self._system_buffer.length(), SAMPLE_RATE, "system"
            )
            mic_rate = (
                self._handler._mic_sample_rate
                if self._handler and self._handler._mic_sample_rate
                else MIC_SAMPLE_RATE
            )
            mic_speech = (
                self._detect_speech_in_buffer(
                    self._mic_buffer, self._mic_buffer.length(), mic_rate, "mic"
                )
                if self.include_mic
                else False
            )

            # Update silence timer
            if system_speech or mic_speech:
                self._last_speech_time = current_time
                return False

            if self._last_speech_time is None:
                self._last_speech_time = current_time
                return False

            return (current_time - self._last_speech_time) >= SILENCE_TIMEOUT

        except Exception as e:
            print(f"  > critical VAD error: {e}")
            return False

    def _save_audio_file(self, samples: np.ndarray, path: Path) -> bool:
        """Save audio samples to file. Returns True if successful."""
        try:
            int16 = float32_to_int16(samples)
            save_fn = save_wav if path.suffix == ".wav" else save_mp3
            size = save_fn(int16, path)
            duration = len(samples) / SAMPLE_RATE
            print(f"  > saved {path.name} ({size // 1024} KB, {duration:.1f}s)")
            return True
        except Exception as e:
            print(f"  > error saving {path.name}: {e}")
            traceback.print_exc()
            return False

    def _stop_stream(self):
        """Stop the capture stream."""
        event = threading.Event()
        self.stream.stopCaptureWithCompletionHandler_(lambda e: event.set())
        event.wait(timeout=5.0)

    def stop(self):
        """Stop recording and process audio."""
        if not self._running and not self.stream:
            return

        self._running = False

        # Stop stream
        if self.stream:
            safe_execute(self._stop_stream, "warning: error stopping stream")
            self.stream = None

        print(LINE)

        # Detect actual mic sample rate (fallback to constant)
        mic_rate = MIC_SAMPLE_RATE
        if self._handler and self._handler._mic_sample_rate:
            mic_rate = self._handler._mic_sample_rate
            if mic_rate != MIC_SAMPLE_RATE:
                print(f"  > mic sample rate: {mic_rate} Hz (detected)")

        # Get audio buffers (critical - must succeed)
        _, system_samples = safe_execute(
            lambda: self._system_buffer.get_samples(), "error reading system buffer", critical=True
        )
        _, mic_samples = safe_execute(
            lambda: self._mic_buffer.get_samples() if self.include_mic else None,
            "error reading mic buffer",
            critical=True,
        )

        if self.mic_only:
            # Mic-only mode: save mic audio as primary output
            if mic_samples is not None and len(mic_samples) > 0:
                mic_resampled = resample(mic_samples, mic_rate, SAMPLE_RATE)
                self._save_audio_file(mic_resampled, self.output_path)
            else:
                print("  > warning: no mic audio captured")
        else:
            # Normal mode: save system audio + optional mic
            if system_samples is not None and len(system_samples) > 0:
                self._save_audio_file(system_samples, self.output_path)
            else:
                print("  > warning: no system audio captured")

        # Save mic audio as separate file (not in mic-only mode, already saved as primary)
        mic_path = None
        if not self.mic_only and mic_samples is not None and len(mic_samples) > 0:
            mic_path = self.output_path.with_stem(f"{self.output_path.stem}_mic")
            mic_resampled = resample(mic_samples, mic_rate, SAMPLE_RATE)
            if not self._save_audio_file(mic_resampled, mic_path):
                mic_path = None
        elif not self.mic_only and self.include_mic:
            print("  > warning: no mic audio captured")

        # Live stats
        if self._transcriber:
            def print_stats():
                stats = self._transcriber.get_stats()
                print(f"  > live: {stats['updates']} updates, {stats['avg_time']:.2f}s avg")
            safe_execute(print_stats, "warning: error getting live stats")

        # Final transcription
        if self.final and self.output_path.exists():
            safe_execute(lambda: self._transcribe_final(mic_path), "error during final transcription", critical=True)

    def _transcribe_final(self, mic_path: Path | None):
        """Run final high-quality transcription."""
        model_name = self.model.split("/")[-1]
        detect_speech = self.mic_only  # Use VAD for mic-only (filters silence)
        print(f"  > transcribing with {model_name} (language: {self.whisper_language})...")

        # Transcribe primary audio
        def transcribe_system():
            result = transcribe_audio(
                self.output_path, self.model, self.whisper_language, detect_speech=detect_speech
            )
            text = format_transcript(result["segments"]) or result["text"]
            output_path = self.output_path.with_suffix(".txt")
            output_path.write_text(text)
            print(LINE)
            print(text)
            print(LINE)
            print(f"  > saved {output_path.name} ({result['duration_seconds']:.1f}s)")
            return text

        success, text = safe_execute(transcribe_system, "error transcribing primary audio", critical=True)

        # Transcribe mic audio (only when not mic-only, since primary is already mic)
        mic_text = None
        if mic_path and mic_path.exists():
            def transcribe_mic():
                print(f"  > transcribing mic with {model_name}...")
                mic_result = transcribe_audio(mic_path, self.model, self.whisper_language, detect_speech=True)
                if mic_result["segments"]:
                    result_text = format_transcript(mic_result["segments"])
                    mic_output = mic_path.with_suffix(".txt")
                    mic_output.write_text(result_text)
                    print(f"  > saved {mic_output.name} ({mic_result['duration_seconds']:.1f}s)")
                    return result_text
                return None

            _, mic_text = safe_execute(transcribe_mic, "error transcribing mic audio", critical=True)

        # Generate summary
        if self.summarize and text:
            def generate_summary():
                from .summarizer import summarize_file
                summary = summarize_file(
                    self.output_path.with_suffix(".txt"),
                    meeting=self.meeting,
                    mic_transcript=mic_text,
                    language=self.language,
                )
                if summary:
                    print(LINE)
                    print(summary)
                    print(LINE)

            safe_execute(generate_summary, "error generating summary", critical=True)

    def _get_shareable_content(self):
        """Get shareable content synchronously."""
        result = {"content": None, "error": None}
        event = threading.Event()

        def handler(content, error):
            result["content"] = content
            result["error"] = error
            event.set()

        SCShareableContent.getShareableContentWithCompletionHandler_(handler)

        if not event.wait(timeout=10.0):
            raise TimeoutError("Timeout getting shareable content")
        if result["error"]:
            raise RuntimeError(f"Failed: {result['error'].localizedDescription()}")

        return result["content"]

    def _start_stream(self):
        """Start capture stream."""
        result = {"error": None}
        event = threading.Event()

        def handler(error):
            result["error"] = error
            event.set()

        self.stream.startCaptureWithCompletionHandler_(handler)

        if not event.wait(timeout=10.0):
            raise TimeoutError("Timeout starting capture")
        if result["error"]:
            raise RuntimeError(f"Failed: {result['error'].localizedDescription()}")
