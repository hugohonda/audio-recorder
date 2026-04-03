# Audio Recorder

Capture macOS system audio + microphone using ScreenCaptureKit. Live transcription, final transcription (MLX Whisper), and AI summarization (Gemini).

## Requirements

- macOS 12.3+, Python 3.12+
- Screen Recording permission (`System Settings > Privacy & Security > Screen Recording`)

## Installation

```bash
git clone <repo-url>
cd audio-recorder
uv sync
```

## Quick Start (Makefile)

```bash
make record              # English, system + mic
make record-mic          # English, mic only
make record-live         # English, live transcription
make record-live-mic     # English, live + mic only

make record-pt           # Portuguese, system + mic
make record-pt-mic       # Portuguese, mic only
make record-pt-live      # Portuguese, live transcription
make record-pt-live-mic  # Portuguese, live + mic only
```

## CLI Usage

```bash
# Record
uv run audio-recorder record [options]
  -d 60           # Duration in seconds
  -o meeting.mp3  # Custom filename
  --no-mic        # System audio only
  --mic-only      # Mic audio only
  --live          # Live transcription
  --no-final      # Skip final transcription
  --no-summary    # Skip AI summary
  -m turbo        # Whisper model (tiny/small/distil/turbo/large)
  -l pt-br        # Language (en/pt-br)

# Transcribe existing file
uv run audio-recorder transcribe recording.mp3 [-m large] [-l pt-br]

# Summarize transcript
uv run audio-recorder summarize transcript.txt [-l pt-br]

# List models
uv run audio-recorder models
```

## Output Files

- `*.mp3` — Audio (system + `*_mic.mp3` for mic)
- `*.txt` — Timestamped transcript
- `*_summary.md` — AI-generated summary

## How It Works

1. **ScreenCaptureKit** captures system audio at 16kHz mono
2. **Microphone** captured at 24kHz, resampled to 16kHz
3. **Live transcription** — Moonshine (English) or Whisper-small (multilingual, 30x realtime)
4. **Final transcription** — MLX Whisper (high-quality)
5. **VAD** — Silero filters mic silence
6. **Meeting detection** — Auto-detects active meeting from `meetings.json` for richer summaries
7. **AI Summary** — Gemini generates structured meeting notes

## Project Structure

```
src/audio_recorder/
├── cli.py         # CLI commands
├── capture.py     # ScreenCaptureKit audio capture
├── live.py        # Live transcription (Moonshine/Whisper)
├── transcribe.py  # Batch transcription utilities
├── audio.py       # Audio processing (buffer, resample, VAD)
├── summarizer.py  # Gemini AI summarization
└── meeting.py     # Meeting auto-detection
```

## Dev

```bash
make lint    # ruff check + format
make test    # pytest
```

## License

MIT
