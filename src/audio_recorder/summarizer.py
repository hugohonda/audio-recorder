"""Meeting summarization using Gemini via Vertex AI."""

import os
import time
from pathlib import Path

from google import genai

_PROMPT_PATH = Path(__file__).parent / "prompts" / "summary.md"
_MODEL = "gemini-2.5-flash"


def _log(msg: str) -> None:
    print(f"  > {msg}")


def summary_path_for(path: Path) -> Path:
    """Derive the summary output path from an audio or transcript path."""
    return path.with_stem(path.stem + "_summary").with_suffix(".md")


_MIC_TRANSCRIPT_SECTIONS = {
    "en": (
        "## Hugo's Microphone Transcript\n"
        "\n"
        "The following transcript was captured directly from Hugo's microphone "
        "during the meeting. It contains only Hugo's voice, recorded in isolation "
        "from other participants.\n"
        "\n"
        "**How to use this:**\n"
        "- Cross-reference these timestamps with the main transcript to identify "
        "which statements are Hugo's.\n"
        "- Attribute statements that match (or closely match) the mic transcript "
        "content to Hugo.\n"
        "- By elimination, attribute other statements to other participants.\n"
        "- The mic transcript timings may not align perfectly with the main "
        "transcript — use content matching as the primary signal.\n"
        "\n"
        "```\n"
        "{mic_transcript}\n"
        "```"
    ),
    "pt-br": (
        "## Transcrição do Microfone do Hugo\n"
        "\n"
        "A seguinte transcrição foi capturada diretamente do microfone do Hugo "
        "durante a reunião. Contém apenas a voz do Hugo, gravada isoladamente "
        "dos demais participantes.\n"
        "\n"
        "**Como usar:**\n"
        "- Cruze os timestamps com a transcrição principal para identificar "
        "quais falas são do Hugo.\n"
        "- Atribua ao Hugo as falas que coincidam (ou sejam muito similares) "
        "com o conteúdo da transcrição do microfone.\n"
        "- Por eliminação, atribua as demais falas aos outros participantes.\n"
        "- Os timestamps do microfone podem não se alinhar perfeitamente com a "
        "transcrição principal — use a correspondência de conteúdo como sinal primário.\n"
        "\n"
        "```\n"
        "{mic_transcript}\n"
        "```"
    ),
}


def summarize(
    transcript: str,
    output_path: Path,
    project: str | None = None,
    location: str | None = None,
    meeting: dict | None = None,
    mic_transcript: str | None = None,
    language: str = "en",
) -> str:
    """Summarize a transcript using Gemini and write the result to output_path.

    Uses Vertex AI with Application Default Credentials.
    If meeting metadata is provided, it is injected into the prompt for context.
    If mic_transcript is provided, it is injected to help with speaker attribution.
    Returns the summary text.
    """
    project = project or os.environ.get("VERTEX_PROJECT", "toptal-agent-ops")
    location = location or os.environ.get("VERTEX_LOCATION", "us-central1")

    # Select prompt based on language
    prompt_file = f"summary_{language}.md" if language == "pt-br" else "summary.md"
    prompt_path = _PROMPT_PATH.parent / prompt_file
    if not prompt_path.exists():
        prompt_path = _PROMPT_PATH  # Fallback to English
    prompt_template = prompt_path.read_text()

    if meeting:
        from .meeting import format_meeting_context

        meeting_context = format_meeting_context(meeting, language=language)
    else:
        meeting_context = ""

    if mic_transcript:
        mic_template = _MIC_TRANSCRIPT_SECTIONS.get(language, _MIC_TRANSCRIPT_SECTIONS["en"])
        mic_section = mic_template.replace("{mic_transcript}", mic_transcript)
    else:
        mic_section = ""

    prompt = prompt_template.replace("{meeting_context}", meeting_context)
    prompt = prompt.replace("{mic_transcript_section}", mic_section)
    prompt = prompt.replace("{transcript}", transcript)

    _log(f"summarizing with {_MODEL}...")

    client = genai.Client(vertexai=True, project=project, location=location)

    start = time.time()
    response = client.models.generate_content(model=_MODEL, contents=prompt)
    elapsed = time.time() - start

    summary = response.text.strip()
    output_path.write_text(summary)
    _log(f"saved {output_path.name} ({elapsed:.1f}s)")

    return summary


def summarize_file(
    transcript_path: Path,
    meeting: dict | None = None,
    mic_transcript: str | None = None,
    language: str = "en",
) -> str | None:
    """Read a transcript file, summarize it, and return the summary text.

    Returns None if the transcript is empty or summarization fails.
    """
    transcript = transcript_path.read_text().strip()
    if not transcript:
        return None

    try:
        return summarize(
            transcript,
            summary_path_for(transcript_path),
            meeting=meeting,
            mic_transcript=mic_transcript,
            language=language,
        )
    except Exception as e:
        _log(f"summarization failed: {e}")
        return None
