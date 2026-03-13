"""CLI entrypoint for audio-recorder."""

import importlib
import time
from datetime import datetime
from pathlib import Path

import click

RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"
MEETINGS_PATH = Path(__file__).parent.parent.parent / "meetings.json"
LINE = "─" * 50

# MLX Whisper models (best to fastest)
WHISPER_MODELS = {
    "large": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "distil": "mlx-community/distil-whisper-large-v3",
    "small": "mlx-community/whisper-small-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def default_output_path() -> str:
    """Generate default output path with timestamp."""
    RECORDINGS_DIR.mkdir(exist_ok=True)
    return str(RECORDINGS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S.mp3"))


class LazyGroup(click.Group):
    """Click group that lazily loads subcommands for faster startup."""

    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        return base + sorted(self.lazy_subcommands.keys())

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.lazy_subcommands:
            import_path = self.lazy_subcommands[cmd_name]
            modname, cmd_obj_name = import_path.rsplit(".", 1)
            mod = importlib.import_module(modname)
            return getattr(mod, cmd_obj_name)
        return super().get_command(ctx, cmd_name)


@click.group(cls=LazyGroup)
@click.version_option()
def main():
    """Capture and transcribe macOS system audio."""


@main.command()
@click.option("-o", "--output", default=None, help="Output file path")
@click.option("-d", "--duration", type=int, help="Recording duration (seconds)")
@click.option("--no-mic", is_flag=True, help="System audio only")
@click.option("--live", is_flag=True, help="Live transcription (Moonshine for en, Whisper for pt-br)")
@click.option("--no-final", is_flag=True, help="Skip Whisper transcription after recording")
@click.option(
    "-m",
    "--model",
    type=click.Choice(list(WHISPER_MODELS.keys())),
    default="distil",
    help="Whisper model for final transcription",
)
@click.option("--no-summary", is_flag=True, help="Skip Gemini summarization")
@click.option(
    "-l",
    "--lang",
    type=click.Choice(["en", "pt-br"]),
    default="en",
    help="Language for transcription (en or pt-br)",
)
def record(
    output: str | None,
    duration: int | None,
    no_mic: bool,
    live: bool,
    no_final: bool,
    model: str,
    no_summary: bool,
    lang: str,
):
    """Record system audio + microphone to MP3."""
    from .capture import AudioRecorder

    output = output or default_output_path()

    # Try to match a meeting from meetings.json
    meeting = None
    if MEETINGS_PATH.exists():
        try:
            from .meeting import find_active_meeting, load_meetings

            meetings = load_meetings(MEETINGS_PATH)
            meeting = find_active_meeting(meetings)
            if meeting:
                click.echo(f"  > meeting detected: {meeting['name']}")
        except Exception as e:
            click.echo(f"  > warning: failed to load meetings.json: {e}")

    # Map pt-br to pt for Whisper
    whisper_lang = "pt" if lang == "pt-br" else lang

    # Distil model doesn't support non-English well, switch to turbo for better results
    if lang != "en" and model == "distil":
        click.echo(f"  > note: switching to 'turbo' model for better {lang} support (distil is English-only)")
        model = "turbo"

    recorder = AudioRecorder(
        output_path=output,
        include_mic=not no_mic,
        live_transcribe=live,
        final_transcribe=not no_final,
        whisper_model=WHISPER_MODELS[model],
        summarize=not no_summary,
        meeting=meeting,
        language=whisper_lang,
    )
    recorder.start(duration=duration)


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "-m",
    "--model",
    type=click.Choice(list(WHISPER_MODELS.keys())),
    default="distil",
    help="Whisper model: large, turbo, distil (default), small, tiny",
)
@click.option("--no-summary", is_flag=True, help="Skip Gemini summarization")
@click.option(
    "-l",
    "--lang",
    type=click.Choice(["en", "pt-br"]),
    default="en",
    help="Language for transcription (en or pt-br)",
)
def transcribe(audio_file: str, model: str, no_summary: bool, lang: str):
    """Transcribe audio file using MLX Whisper."""
    import mlx_whisper

    audio_path = Path(audio_file)
    output_path = audio_path.with_suffix(".txt")

    # Map pt-br to pt for Whisper
    whisper_lang = "pt" if lang == "pt-br" else lang

    # Distil model doesn't support non-English well, switch to turbo for better results
    if lang != "en" and model == "distil":
        click.echo(f"  > note: switching to 'turbo' model for better {lang} support (distil is English-only)")
        model = "turbo"

    model_path = WHISPER_MODELS[model]
    model_short = model_path.split("/")[-1]

    click.echo(f"\naudio-recorder | transcribe {audio_path.name}")
    click.echo(f"  > model: {model_short}")
    click.echo(f"  > language: {lang}")

    from .audio import (
        detect_speech_segments,
        filter_segments_by_speech,
        format_segments,
    )

    start = time.time()
    result = mlx_whisper.transcribe(
        str(audio_path), path_or_hf_repo=model_path, language=whisper_lang, task="transcribe"
    )
    elapsed = time.time() - start

    text = format_segments(result.get("segments", []))
    if not text:
        text = result["text"].strip()
    output_path.write_text(text)

    click.echo(LINE)
    click.echo(text)
    click.echo(LINE)
    click.echo(f"  > saved {output_path.name} ({elapsed:.1f}s)")

    # Transcribe mic audio if it exists
    mic_path = audio_path.with_stem(audio_path.stem + "_mic")
    mic_transcript = None
    if mic_path.exists():
        mic_txt = mic_path.with_suffix(".txt")

        click.echo("  > detecting speech in mic audio...")
        speech_ranges = detect_speech_segments(mic_path)
        if not speech_ranges:
            click.echo("  > no speech detected in mic audio, skipping")
        else:
            click.echo(f"  > transcribing mic with {model_short}...")
            mic_start = time.time()
            mic_result = mlx_whisper.transcribe(
                str(mic_path),
                path_or_hf_repo=model_path,
                condition_on_previous_text=False,
                language=whisper_lang,
                task="transcribe",
            )
            mic_elapsed = time.time() - mic_start

            segments = mic_result.get("segments", [])
            segments = filter_segments_by_speech(segments, speech_ranges)
            mic_text = format_segments(segments)
            if mic_text:
                mic_txt.write_text(mic_text)
                click.echo(f"  > saved {mic_txt.name} ({mic_elapsed:.1f}s)")
                mic_transcript = mic_text

    if not no_summary and text:
        from .summarizer import summarize_file

        summary = summarize_file(output_path, mic_transcript=mic_transcript)
        if summary:
            click.echo(LINE)
            click.echo(summary)
            click.echo(LINE)


@main.command()
@click.argument("transcript_file", type=click.Path(exists=True))
@click.option(
    "--meeting",
    "meeting_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to meetings.json for context",
)
def summarize(transcript_file: str, meeting_path: str | None):
    """Summarize a transcript file using Gemini."""
    from .summarizer import summarize_file

    transcript_path = Path(transcript_file)
    click.echo(f"\naudio-recorder | summarize {transcript_path.name}")

    meeting = None
    if meeting_path:
        try:
            from .meeting import find_active_meeting, load_meetings

            meetings = load_meetings(meeting_path)
            meeting = find_active_meeting(meetings)
            if meeting:
                click.echo(f"  > meeting context: {meeting['name']}")
            else:
                click.echo("  > no matching meeting found for current time")
        except Exception as e:
            click.echo(f"  > warning: failed to load meeting file: {e}")

    # Check for mic transcript
    mic_transcript = None
    mic_txt = transcript_path.with_stem(
        transcript_path.stem.replace("_mic", "") + "_mic"
    ).with_suffix(".txt")
    if mic_txt.exists() and mic_txt != transcript_path:
        mic_transcript = mic_txt.read_text().strip() or None
        if mic_transcript:
            click.echo(f"  > using mic transcript: {mic_txt.name}")

    summary = summarize_file(transcript_path, meeting=meeting, mic_transcript=mic_transcript)
    if not summary:
        click.echo("  > transcript is empty or summarization failed")
        return

    click.echo(LINE)
    click.echo(summary)
    click.echo(LINE)


@main.command()
def models():
    """List available transcription models."""
    descs = {
        "large": "Max accuracy, slowest",
        "turbo": "Great accuracy, fast",
        "distil": "Best balance (default) - English only",
        "small": "Faster, decent accuracy",
        "tiny": "Fastest, lower accuracy",
    }
    click.echo("Available Whisper models (--model/-m):\n")
    for name, path in WHISPER_MODELS.items():
        click.echo(f"  {name:8} - {descs[name]}")
        click.echo(f"           {path}\n")
    click.echo("Note: For non-English languages (--lang pt-br), use 'turbo' or 'large'.")


if __name__ == "__main__":
    main()
