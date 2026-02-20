"""Meeting summarization using Gemini via Vertex AI."""

import time
from pathlib import Path

from google import genai

_PROMPT_PATH = Path(__file__).parent / "prompts" / "summary.md"
_MODEL = "gemini-2.5-flash"


def _log(msg: str) -> None:
    print(f"  > {msg}")


def summarize(
    transcript: str,
    output_path: Path,
    project: str = "toptal-agent-ops",
    location: str = "us-central1",
) -> str:
    """Summarize a transcript using Gemini and write the result to output_path.

    Uses Vertex AI with Application Default Credentials.
    Returns the summary text.
    """
    prompt_template = _PROMPT_PATH.read_text()
    prompt = prompt_template.replace("{transcript}", transcript)

    _log(f"summarizing with {_MODEL}...")

    client = genai.Client(vertexai=True, project=project, location=location)

    start = time.time()
    response = client.models.generate_content(model=_MODEL, contents=prompt)
    elapsed = time.time() - start

    summary = response.text.strip()
    output_path.write_text(summary)
    _log(f"saved {output_path.name} ({elapsed:.1f}s)")

    return summary
