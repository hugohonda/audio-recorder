"""Audio capture using macOS ScreenCaptureKit."""

import sys
import threading
import time
from pathlib import Path

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
    resample,
    save_mp3,
    save_wav,
)
from .transcribe import format_transcript, transcribe_audio

LINE = "─" * 50


class AudioStreamOutput(NSObject):
    """Handle incoming audio from ScreenCaptureKit."""

    def initWithSystemBuffer_micBuffer_(self, system_buffer, mic_buffer):
        self = objc.super(AudioStreamOutput, self).init()
        if self is None:
            return None
        self._system_buffer = system_buffer
        self._mic_buffer = mic_buffer
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
        live: bool = False,
        final: bool = True,
        model: str = "mlx-community/distil-whisper-large-v3",
        summarize: bool = True,
        language: str = "en",
    ):
        self.output_path = Path(output_path)
        self.include_mic = include_mic
        self.live = live
        self.final = final
        self.model = model
        self.summarize = summarize
        self.language = language

        self.stream = None
        self._system_buffer = None
        self._mic_buffer = None
        self._running = False
        self._transcriber = None

        # Setup live transcription
        if live:
            from .live import LiveTranscriber

            live_path = self.output_path.with_stem(f"{self.output_path.stem}_live").with_suffix(".txt")
            self._transcriber = LiveTranscriber(language=language, output_path=live_path)

    def start(self, duration: int | None = None):
        """Start recording."""
        self._system_buffer = AudioBuffer()
        self._mic_buffer = AudioBuffer()

        # Header
        mode = "system + mic" if self.include_mic else "system only"
        parts = [mode]
        if self._transcriber:
            engine = "moonshine" if self.language == "en" else f"whisper-tiny ({self.language})"
            parts.append(f"live ({engine})")
        if self.final:
            parts.append(self.model.split("/")[-1])

        print(f"\naudio-recorder | {', '.join(parts)}")

        # Preload live model
        if self._transcriber:
            model_name = "moonshine" if self.language == "en" else f"whisper-tiny ({self.language})"
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

        handler = AudioStreamOutput.alloc().initWithSystemBuffer_micBuffer_(
            self._system_buffer, self._mic_buffer
        )
        self.stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, handler
        )

        self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
            handler, SCStreamOutputTypeAudio, None, objc.nil
        )
        if self.include_mic:
            self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
                handler, SCStreamOutputTypeMicrophone, None, objc.nil
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
        interval = self._transcriber.update_interval if self._transcriber else 3.0

        while self._running:
            time.sleep(0.1)
            now = time.monotonic()

            if duration and (now - start) >= duration:
                break

            if self._transcriber and (now - last_update) >= interval:
                text, _ = self._transcriber.process_buffer(
                    self._system_buffer.get_range_np,
                    self._system_buffer.length(),
                )
                if text:
                    print(text)
                last_update = now

        # Final live update
        if self._transcriber:
            text, _ = self._transcriber.process_buffer_sync(
                self._system_buffer.get_range_np,
                self._system_buffer.length(),
            )
            if text:
                print(text)

    def stop(self):
        """Stop recording and process audio."""
        if not self._running and not self.stream:
            return

        self._running = False

        if self.stream:
            event = threading.Event()
            self.stream.stopCaptureWithCompletionHandler_(lambda e: event.set())
            event.wait(timeout=5.0)
            self.stream = None

        print(LINE)

        # Save audio files
        system_samples = self._system_buffer.get_samples()
        mic_samples = self._mic_buffer.get_samples() if self.include_mic else None

        if len(system_samples) > 0:
            int16 = float32_to_int16(system_samples)
            save_fn = save_wav if self.output_path.suffix == ".wav" else save_mp3
            size = save_fn(int16, self.output_path)
            duration = len(system_samples) / SAMPLE_RATE
            print(f"  > saved {self.output_path.name} ({size // 1024} KB, {duration:.1f}s)")

        mic_path = None
        if mic_samples is not None and len(mic_samples) > 0:
            mic_path = self.output_path.with_stem(f"{self.output_path.stem}_mic")
            mic_resampled = resample(mic_samples, MIC_SAMPLE_RATE, SAMPLE_RATE)
            int16 = float32_to_int16(mic_resampled)
            save_fn = save_wav if self.output_path.suffix == ".wav" else save_mp3
            size = save_fn(int16, mic_path)
            duration = len(mic_resampled) / SAMPLE_RATE
            print(f"  > saved {mic_path.name} ({size // 1024} KB, {duration:.1f}s)")

        # Live stats
        if self._transcriber:
            stats = self._transcriber.get_stats()
            print(f"  > live: {stats['updates']} updates, {stats['avg_time']:.2f}s avg")

        # Final transcription
        if self.final and self.output_path.exists():
            self._transcribe_final(mic_path)

    def _transcribe_final(self, mic_path: Path | None):
        """Run final high-quality transcription."""
        model_name = self.model.split("/")[-1]
        print(f"  > transcribing with {model_name} (language: {self.language})...")

        result = transcribe_audio(self.output_path, self.model, self.language)
        text = format_transcript(result["segments"]) or result["text"]

        output_path = self.output_path.with_suffix(".txt")
        output_path.write_text(text)

        print(LINE)
        print(text)
        print(LINE)
        print(f"  > saved {output_path.name} ({result['duration_seconds']:.1f}s)")

        # Mic transcription
        mic_text = None
        if mic_path and mic_path.exists():
            print(f"  > transcribing mic with {model_name}...")
            mic_result = transcribe_audio(mic_path, self.model, self.language, detect_speech=True)

            if mic_result["segments"]:
                mic_text = format_transcript(mic_result["segments"])
                mic_output = mic_path.with_suffix(".txt")
                mic_output.write_text(mic_text)
                print(f"  > saved {mic_output.name} ({mic_result['duration_seconds']:.1f}s)")

        # Summary
        if self.summarize and text:
            from .summarizer import summarize_file

            summary = summarize_file(output_path, mic_transcript=mic_text)
            if summary:
                print(LINE)
                print(summary)
                print(LINE)

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
