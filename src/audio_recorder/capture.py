"""Audio capture using ScreenCaptureKit."""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import objc
from CoreMedia import (
    CMBlockBufferCopyDataBytes,
    CMBlockBufferGetDataLength,
    CMSampleBufferGetDataBuffer,
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
    format_segments,
    resample,
    save_mp3,
    save_wav,
)

LINE = "─" * 50


def _log(msg: str) -> None:
    """Print a system log line with prefix."""
    print(f"  > {msg}")


class AudioStreamOutput(NSObject):
    """Output handler for audio stream from ScreenCaptureKit."""

    def initWithSystemBuffer_micBuffer_(self, system_buffer, mic_buffer):
        self = objc.super(AudioStreamOutput, self).init()
        if self is None:
            return None
        self._system_buffer = system_buffer
        self._mic_buffer = mic_buffer
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
        """Handle incoming audio sample buffer."""
        if output_type not in (SCStreamOutputTypeAudio, SCStreamOutputTypeMicrophone):
            return

        try:
            block_buffer = CMSampleBufferGetDataBuffer(sample_buffer)
            if block_buffer is None:
                return

            data_length = CMBlockBufferGetDataLength(block_buffer)
            if data_length == 0:
                return

            status, audio_data = CMBlockBufferCopyDataBytes(block_buffer, 0, data_length, None)
            if status != 0:
                return

            audio_bytes = bytes(audio_data)
            if output_type == SCStreamOutputTypeAudio:
                self._system_buffer.add(audio_bytes)
            else:
                self._mic_buffer.add(audio_bytes)

        except Exception as e:
            _log(f"error: {e}")

    def stream_didStopWithError_(self, stream, error):
        """Handle stream stop."""
        if error:
            _log(f"stream error: {error.localizedDescription()}")


class AudioRecorder:
    """Records system audio + microphone using ScreenCaptureKit."""

    def __init__(
        self,
        output_path: str,
        include_mic: bool = True,
        live_transcribe: bool = False,
        final_transcribe: bool = True,
        whisper_model: str = "mlx-community/distil-whisper-large-v3",
        summarize: bool = True,
    ):
        self.output_path = Path(output_path)
        self.include_mic = include_mic
        self.live_transcribe = live_transcribe
        self.final_transcribe = final_transcribe
        self.whisper_model = whisper_model
        self.summarize = summarize

        self.stream: SCStream | None = None
        self.output_handler: AudioStreamOutput | None = None
        self._system_buffer: AudioBuffer | None = None
        self._mic_buffer: AudioBuffer | None = None
        self._is_running = False

        # Moonshine for live transcription
        self._transcriber = None
        if live_transcribe:
            from .moonshine_transcriber import MoonshineTranscriber

            live_path = self._live_transcript_path()
            self._transcriber = MoonshineTranscriber(
                output_path=live_path,
            )

    def _live_transcript_path(self) -> Path:
        """Path for live transcript file."""
        return self.output_path.with_stem(self.output_path.stem + "_live").with_suffix(".txt")

    def _final_transcript_path(self) -> Path:
        """Path for final Whisper transcript file."""
        return self.output_path.with_suffix(".txt")

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

    def _get_content_filter(self) -> SCContentFilter:
        """Get a content filter for capturing system audio."""
        content = self._get_shareable_content()
        displays = content.displays()
        if not displays:
            raise RuntimeError("No displays found")
        return SCContentFilter.alloc().initWithDisplay_excludingWindows_(displays[0], [])

    def _configure_stream(self) -> SCStreamConfiguration:
        """Configure the audio stream."""
        config = SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setExcludesCurrentProcessAudio_(False)
        if self.include_mic:
            config.setCaptureMicrophone_(True)
        config.setWidth_(2)
        config.setHeight_(2)
        config.setSampleRate_(SAMPLE_RATE)
        config.setChannelCount_(CHANNELS)
        return config

    def _start_stream(self):
        """Start the capture stream."""
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

    def _stop_stream(self):
        """Stop the capture stream."""
        if not self.stream:
            return
        event = threading.Event()
        self.stream.stopCaptureWithCompletionHandler_(lambda e: event.set())
        event.wait(timeout=5.0)

    def _save_audio(self) -> None:
        """Save system audio and microphone to separate files."""
        system_samples = self._system_buffer.get_samples()
        mic_samples = self._mic_buffer.get_samples() if self.include_mic else np.array([])

        sys_duration = len(system_samples) / SAMPLE_RATE

        if len(mic_samples) > 0:
            mic_resampled = resample(mic_samples, MIC_SAMPLE_RATE, SAMPLE_RATE)
        else:
            mic_resampled = np.array([])

        # Save system audio
        if len(system_samples) > 0:
            int16 = float32_to_int16(system_samples)
            if self.output_path.suffix.lower() == ".wav":
                size = save_wav(int16, self.output_path)
            else:
                size = save_mp3(int16, self.output_path)
            _log(f"saved {self.output_path.name} ({size // 1024} KB, {sys_duration:.1f}s)")
        else:
            _log("no system audio captured")

        # Save mic audio to separate file
        if len(mic_resampled) > 0:
            mic_path = self.output_path.with_stem(self.output_path.stem + "_mic")
            int16_mic = float32_to_int16(mic_resampled)
            mic_dur = len(mic_resampled) / SAMPLE_RATE
            if self.output_path.suffix.lower() == ".wav":
                size = save_wav(int16_mic, mic_path)
            else:
                size = save_mp3(int16_mic, mic_path)
            _log(f"saved {mic_path.name} ({size // 1024} KB, {mic_dur:.1f}s)")

    def start(self, duration: int | None = None):
        """Start recording audio."""
        self._system_buffer = AudioBuffer()
        self._mic_buffer = AudioBuffer()

        # Header
        mode = "system + mic" if self.include_mic else "system only"
        features = [mode]
        if self._transcriber:
            features.append("live transcription")
        if self.final_transcribe:
            features.append(self.whisper_model.split("/")[-1])
        print(f"\naudio-recorder | {', '.join(features)}")

        # Preload Moonshine before recording
        if self._transcriber:
            sys.stdout.write(f"  > loading {self._transcriber.model_name}... ")
            sys.stdout.flush()
            load_time = self._transcriber.load_model(quiet=True)
            print(f"ready ({load_time:.1f}s)")

        config = self._configure_stream()
        content_filter = self._get_content_filter()

        self.output_handler = AudioStreamOutput.alloc().initWithSystemBuffer_micBuffer_(
            self._system_buffer, self._mic_buffer
        )
        self.stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, self.output_handler
        )

        # Add outputs
        self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self.output_handler, SCStreamOutputTypeAudio, None, objc.nil
        )
        if self.include_mic:
            self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self.output_handler,
                SCStreamOutputTypeMicrophone,
                None,
                objc.nil,
            )

        self._is_running = True
        self._start_stream()

        _log(f"recording to {self.output_path.name} (ctrl+c to stop)")
        print(LINE)

        try:
            self._run_loop(duration)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _run_loop(self, duration: int | None):
        """Main recording loop with live transcription."""
        start_time = time.monotonic()
        last_transcribe = start_time
        interval = self._transcriber.update_interval if self._transcriber else 3.0

        while self._is_running:
            time.sleep(0.1)
            now = time.monotonic()
            elapsed = now - start_time

            if duration and elapsed >= duration:
                break

            if self._transcriber and now - last_transcribe >= interval:
                self._process_live_transcription()
                last_transcribe = now

        # Final live transcription update (synchronous to capture last audio)
        if self._transcriber:
            self._process_live_transcription(sync=True)

    def _process_live_transcription(self, *, sync: bool = False):
        """Run Moonshine on current buffer and print new words."""
        if not self._transcriber:
            return

        buffer_len = self._system_buffer.length()
        if buffer_len < SAMPLE_RATE:
            return

        if sync:
            text, _ = self._transcriber.process_buffer_sync(
                self._system_buffer.get_range_np,
                buffer_len,
            )
        else:
            text, _ = self._transcriber.process_buffer(
                self._system_buffer.get_range_np,
                buffer_len,
            )

        if text:
            print(text)

    def stop(self):
        """Stop recording, save audio, run final transcription."""
        if not self._is_running and self.stream is None:
            return

        self._is_running = False

        if self.stream:
            self._stop_stream()
            self.stream = None

        print(LINE)

        if self._system_buffer:
            self._save_audio()

        # Show live transcript stats
        if self._transcriber:
            stats = self._transcriber.get_stats()
            _log(
                f"live transcript: {stats['updates']} updates, {stats['avg_process_time']:.2f}s avg"
            )

        # Auto-transcribe with Whisper for high-quality final transcript
        if self.final_transcribe and self.output_path.exists():
            self._run_final_transcription()

        # Summarize transcript with Gemini
        if self.summarize:
            transcript_path = self._final_transcript_path()
            if transcript_path.exists():
                self._run_summary(transcript_path)

    def _run_final_transcription(self):
        """Run MLX Whisper on the saved audio for high-quality transcript."""
        import mlx_whisper

        output_txt = self._final_transcript_path()
        model_short = self.whisper_model.split("/")[-1]
        _log(f"transcribing with {model_short}...")

        start = time.time()
        result = mlx_whisper.transcribe(
            str(self.output_path),
            path_or_hf_repo=self.whisper_model,
        )
        elapsed = time.time() - start

        text = format_segments(result.get("segments", []))
        if not text:
            text = result["text"].strip()
        output_txt.write_text(text)

        print(LINE)
        print(text)
        print(LINE)
        _log(f"saved {output_txt.name} ({elapsed:.1f}s)")

    def _run_summary(self, transcript_path: Path):
        """Summarize transcript using Gemini."""
        from .summarizer import summarize_file

        summary = summarize_file(transcript_path)
        if summary:
            print(LINE)
            print(summary)
            print(LINE)
