"""Simple CLI for audio recording and transcription."""

from datetime import datetime
from pathlib import Path

import click

from .transcribe import format_transcript, get_best_model_for_language, transcribe_audio

# Output directory
RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Available models
MODELS = {
    "large": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "distil": "mlx-community/distil-whisper-large-v3",
    "small": "mlx-community/whisper-small-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


@click.group()
@click.version_option()
def cli():
    """Record and transcribe macOS audio."""


@cli.command()
@click.option("-o", "--output", help="Output file path")
@click.option("-d", "--duration", type=int, help="Duration in seconds")
@click.option("--no-mic", is_flag=True, help="System audio only")
@click.option("--live", is_flag=True, help="Live transcription")
@click.option("--no-final", is_flag=True, help="Skip final transcription")
@click.option("-m", "--model", type=click.Choice(list(MODELS.keys())), default="distil")
@click.option("--no-summary", is_flag=True, help="Skip summary")
@click.option("-l", "--lang", type=click.Choice(["en", "pt-br"]), default="en")
def record(output, duration, no_mic, live, no_final, model, no_summary, lang):
    """Record system audio."""
    from .capture import AudioRecorder

    output = output or str(RECORDINGS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S.mp3"))
    language = "pt" if lang == "pt-br" else lang
    model_path = get_best_model_for_language(language, MODELS[model])

    recorder = AudioRecorder(
        output_path=output,
        include_mic=not no_mic,
        live=live,
        final=not no_final,
        model=model_path,
        summarize=not no_summary,
        language=language,
    )
    recorder.start(duration)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("-m", "--model", type=click.Choice(list(MODELS.keys())), default="distil")
@click.option("--no-summary", is_flag=True, help="Skip summary")
@click.option("-l", "--lang", type=click.Choice(["en", "pt-br"]), default="en")
def transcribe(audio_file, model, no_summary, lang):
    """Transcribe an audio file."""
    audio_path = Path(audio_file)
    language = "pt" if lang == "pt-br" else lang
    model_path = get_best_model_for_language(language, MODELS[model])

    click.echo(f"\nTranscribing: {audio_path.name}")
    click.echo(f"  Model: {model_path.split('/')[-1]}")
    click.echo(f"  Language: {lang}")

    # Main audio
    result = transcribe_audio(audio_path, model_path, language)
    text = format_transcript(result["segments"]) or result["text"]

    output_path = audio_path.with_suffix(".txt")
    output_path.write_text(text)

    click.echo(f"\n{text}\n")
    click.echo(f"Saved: {output_path.name} ({result['duration_seconds']:.1f}s)")

    # Mic audio if exists
    mic_path = audio_path.with_stem(f"{audio_path.stem}_mic")
    mic_text = None
    if mic_path.exists():
        click.echo(f"\nTranscribing mic: {mic_path.name}")
        mic_result = transcribe_audio(mic_path, model_path, language, detect_speech=True)
        if mic_result["segments"]:
            mic_text = format_transcript(mic_result["segments"])
            mic_output = mic_path.with_suffix(".txt")
            mic_output.write_text(mic_text)
            click.echo(f"Saved: {mic_output.name}")

    # Summary
    if not no_summary and text:
        from .summarizer import summarize_file

        summary = summarize_file(output_path, mic_transcript=mic_text)
        if summary:
            click.echo(f"\n{summary}\n")


@cli.command()
@click.argument("transcript_file", type=click.Path(exists=True))
def summarize(transcript_file):
    """Summarize a transcript."""
    from .summarizer import summarize_file

    transcript_path = Path(transcript_file)
    click.echo(f"\nSummarizing: {transcript_path.name}")

    # Check for mic transcript
    mic_path = transcript_path.with_stem(f"{transcript_path.stem}_mic").with_suffix(".txt")
    mic_text = mic_path.read_text() if mic_path.exists() else None

    summary = summarize_file(transcript_path, mic_transcript=mic_text)
    if summary:
        click.echo(f"\n{summary}\n")
    else:
        click.echo("No summary generated (empty transcript?)")


@cli.command()
def models():
    """List available models."""
    click.echo("Available Whisper models:\n")
    descs = {
        "large": "Best accuracy, slowest",
        "turbo": "Great accuracy, fast",
        "distil": "Balanced (default, English-only)",
        "small": "Faster",
        "tiny": "Fastest",
    }
    for name, path in MODELS.items():
        click.echo(f"  {name:8} - {descs[name]}")

    click.echo("\nNote: Use 'turbo' or 'large' for non-English languages.")


if __name__ == "__main__":
    cli()
