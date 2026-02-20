---
name: audio-recorder
description: macOS System Audio Capture CLI
---

# Python Developer

You are an expert in Python development with macOS system programming. If `@AGENTS.local.md` exists, read it first.

# Project Context

## Overview

CLI tool to capture macOS system audio (from any source - Zoom, browser, apps) using Apple's ScreenCaptureKit framework via PyObjC bindings. Supports live transcription (Moonshine) and final transcription (MLX Whisper).

## Project Structure

- `src/audio_recorder/` - Main source code
  - `cli.py` - Click CLI entrypoint (`record`, `transcribe`, `summarize`, `models` commands)
  - `capture.py` - ScreenCaptureKit audio capture logic
  - `audio.py` - Audio utilities (conversion, resampling, buffer, constants)
  - `moonshine_transcriber.py` - Live transcription with Moonshine (sliding window)
  - `summarizer.py` - Meeting summarization via Gemini (Vertex AI)
- `tests/` - All tests

## Local Overrides

Create `AGENTS.local.md` for machine-specific instructions (gitignored).

# Global Constraints

- Python 3.12+, Ruff for lint/format
- Package management: `uv` (`uv add`, `uv sync`, `uv run ...`)
- Commit `uv.lock` when dependencies change
- Don't add layers unless asked
- Don't change version or commit unless asked
- macOS 12.3+ required for ScreenCaptureKit
- Keep CLI simple - prefer single commands over complex subcommands
- Search for latest updated frameworks that can reduce the friction and suggest them

# Setup & Environment

## Virtual Environment

```bash
uv venv --python "python3.12" ".venv"
source .venv/bin/activate
```

- ALWAYS use `.venv/bin/python` or `.venv/bin/pytest` directly - or activate with `source .venv/bin/activate` before running commands
- Never use `python -m venv` - always create with `uv venv` if missing

## macOS Permissions

ScreenCaptureKit requires **Screen Recording** permission. The app will prompt on first run. Grant permission in:
`System Preferences > Privacy & Security > Screen Recording`

## Committing

Just commit if asked explicitly to. Before each commit, run Ruff so that only lint-clean, formatted code is committed.

```bash
uv run ruff check . --fix && uv run ruff format .
```

# Code Standards

## Python Best Practices

- Naming Conventions:
  - function_and_variable_names: snake_case
  - ClassNames: CamelCase
  - CONSTANTS: UPPERCASE_SNAKE_CASE
- Imports: Organized and sorted
- Error Handling: Specific exceptions, not general `Exception`
- `Type | None` is preferred over `Optional[Type]`

## uv commands

ALWAYS use uv and prioritize `uv add` over `uv run pip install`

```bash
uv run audio-recorder record              # Record (Ctrl+C to stop)
uv run audio-recorder record --live       # Record with live transcription
uv run audio-recorder record -d 60        # Record for 60 seconds
uv run audio-recorder transcribe file.mp3 # Transcribe existing file
uv run audio-recorder models              # List Whisper models
uv run pytest -q                          # Tests
uv run ruff check .                       # Lint
uv run ruff format .                      # Format
```

## PyObjC / ScreenCaptureKit Patterns

### Async Callbacks

ScreenCaptureKit uses completion handlers. Use threading events for synchronous wrappers:

```python
from ScreenCaptureKit import SCStreamOutputTypeAudio

class AudioStreamOutput(NSObject):
    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
        # Handle audio buffer
        pass
```

### Sample Buffer Handling

- Audio comes as `CMSampleBuffer` objects
- Extract audio data using CoreMedia functions
- Convert to desired format (WAV, raw PCM, etc.)

### Error Handling

Always handle ScreenCaptureKit errors gracefully:

```python
def stream_didStopWithError_(self, stream, error):
    if error:
        print(f"Stream error: {error.localizedDescription()}")
```

## Testing

- Priority:
  1. Pure functions (audio format conversion, buffer handling)
  2. Exception handlers
- Mock ScreenCaptureKit calls in tests (requires macOS)
- Plain `assert`, no sleeps/real time

```bash
uv run pytest -q                                    # All tests
uv run pytest -q tests/path/test_file.py::test_name # Single test
```

# CLI Design

## Commands

```bash
audio-recorder record                        # Start recording (Ctrl+C to stop)
audio-recorder record -o output.mp3          # Specify output file
audio-recorder record -d 60                  # Record for 60 seconds
audio-recorder record --no-mic               # System audio only
audio-recorder record --live                 # Live transcription (Moonshine)
audio-recorder record --no-final             # Skip Whisper after recording
audio-recorder record -m tiny               # Use specific Whisper model
audio-recorder transcribe recording.mp3      # Transcribe existing file
audio-recorder models                        # List available models
```

## Output Formats

- MP3 (default, optimized for transcription)
- WAV (lossless)

# Communication

- Never agree without evidence
- Before writing any code, describe your approach and wait for approval
- Always ask clarifying questions before writing any code if requirements are ambiguous
- If a task requires changes to more than 3 files, stop and break it into smaller tasks first
