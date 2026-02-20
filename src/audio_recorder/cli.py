"""CLI entrypoint for audio-recorder."""

import importlib
import time
from datetime import datetime
from pathlib import Path

import click

RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"
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
@click.option("--live", is_flag=True, help="Live transcription (Moonshine)")
@click.option("--no-final", is_flag=True, help="Skip Whisper transcription after recording")
@click.option(
    "-m",
    "--model",
    type=click.Choice(list(WHISPER_MODELS.keys())),
    default="distil",
    help="Whisper model for final transcription",
)
@click.option("--no-summary", is_flag=True, help="Skip Gemini summarization")
def record(
    output: str | None,
    duration: int | None,
    no_mic: bool,
    live: bool,
    no_final: bool,
    model: str,
    no_summary: bool,
):
    """Record system audio + microphone to MP3."""
    from .capture import AudioRecorder

    output = output or default_output_path()

    recorder = AudioRecorder(
        output_path=output,
        include_mic=not no_mic,
        live_transcribe=live,
        final_transcribe=not no_final,
        whisper_model=WHISPER_MODELS[model],
        summarize=not no_summary,
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
def transcribe(audio_file: str, model: str, no_summary: bool):
    """Transcribe audio file using MLX Whisper."""
    import mlx_whisper

    audio_path = Path(audio_file)
    output_path = audio_path.with_suffix(".txt")
    model_path = WHISPER_MODELS[model]
    model_short = model_path.split("/")[-1]

    click.echo(f"\naudio-recorder | transcribe {audio_path.name}")
    click.echo(f"  > model: {model_short}")

    start = time.time()
    result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model_path)
    elapsed = time.time() - start

    text = result["text"].strip()
    output_path.write_text(text)

    click.echo(LINE)
    click.echo(text)
    click.echo(LINE)
    click.echo(f"  > saved {output_path.name} ({elapsed:.1f}s)")

    if not no_summary and text:
        from .summarizer import summarize_file

        summary = summarize_file(output_path)
        if summary:
            click.echo(LINE)
            click.echo(summary)
            click.echo(LINE)


@main.command()
@click.argument("transcript_file", type=click.Path(exists=True))
def summarize(transcript_file: str):
    """Summarize a transcript file using Gemini."""
    from .summarizer import summarize_file

    transcript_path = Path(transcript_file)
    click.echo(f"\naudio-recorder | summarize {transcript_path.name}")

    summary = summarize_file(transcript_path)
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
        "distil": "Best balance (default)",
        "small": "Faster, decent accuracy",
        "tiny": "Fastest, lower accuracy",
    }
    click.echo("Available Whisper models (--model/-m):\n")
    for name, path in WHISPER_MODELS.items():
        click.echo(f"  {name:8} - {descs[name]}")
        click.echo(f"           {path}\n")


if __name__ == "__main__":
    main()
