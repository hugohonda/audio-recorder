# Audio Recorder

Capture macOS system audio (Zoom calls, YouTube, Spotify) + microphone using ScreenCaptureKit. Live transcription with automatic summarization.

**Features:**
- 🎙️ System audio + microphone recording
- 🌍 Multilingual (English, Portuguese, Spanish, etc.)
- ⚡ Real-time transcription (Moonshine/Whisper)
- 📝 High-quality final transcription (MLX Whisper)
- 🤖 AI summarization (Gemini)

## Requirements

- macOS 12.3+ (Monterey or later)
- Python 3.12+
- Screen Recording permission

## Installation

```bash
git clone <repo-url>
cd audio-recorder
uv sync
```

## Usage

### Record

```bash
# Basic recording (saves to recordings/YYYY-MM-DD_HH-MM-SS.mp3)
uv run audio-recorder record

# Portuguese with live transcription
uv run audio-recorder record --live --lang pt-br

# English with live transcription (faster)
uv run audio-recorder record --live

# Options
uv run audio-recorder record \
  -d 60           # Duration in seconds
  -o meeting.mp3  # Custom filename
  --no-mic        # System audio only
  --no-final      # Skip final transcription
  --no-summary    # Skip AI summary
  -m turbo        # Whisper model (tiny/small/distil/turbo/large)
  -l pt-br        # Language (en/pt-br)
```

### Transcribe Existing Files

```bash
uv run audio-recorder transcribe recording.mp3
uv run audio-recorder transcribe recording.mp3 -m large -l pt-br
```

### Summarize Transcript

```bash
uv run audio-recorder summarize transcript.txt
```

### List Models

```bash
uv run audio-recorder models
```

**Output files:**
- `*.mp3` - Audio (system + mic_mic.mp3)
- `*.txt` - Transcript with timestamps
- `*_summary.md` - AI-generated summary

## First Run

On first run, macOS will prompt for **Screen Recording** permission:

`System Settings > Privacy & Security > Screen Recording`

## How It Works

1. **ScreenCaptureKit** captures system audio at 16kHz mono
2. **Microphone** captured at 24kHz, resampled to 16kHz
3. **Live transcription** (optional):
   - English: Moonshine (fastest, English-optimized)
   - Portuguese/Other: Whisper-tiny (24x realtime, multilingual)
4. **Final transcription**: MLX Whisper (high-quality)
5. **Smart VAD**: Filters mic audio to remove silence
6. **AI Summary**: Gemini generates structured meeting notes

**Audio Format:**
- 16kHz mono MP3 @ 64kbps (~480KB/min)
- Optimized for speech transcription

## Project Structure

```
src/audio_recorder/
├── cli.py         # CLI commands
├── capture.py     # ScreenCaptureKit audio capture
├── live.py        # Live transcription (auto-selects engine)
├── transcribe.py  # Whisper transcription utilities
├── audio.py       # Audio processing (buffer, resample, VAD)
├── summarizer.py  # Gemini AI summarization
└── meeting.py     # Meeting context (optional)
```

## Key Dependencies

- `pyobjc-framework-ScreenCaptureKit` - macOS audio capture
- `mlx-whisper` - Transcription (MLX Whisper)
- `useful-moonshine-onnx` - Live English transcription
- `google-genai` - AI summarization (Gemini)
- `silero-vad` - Voice activity detection
- `lameenc` - MP3 encoding

## Languages Supported

Live transcription:
- 🇺🇸 English (Moonshine - fastest)
- 🇧🇷 Portuguese (Whisper-tiny - 24x realtime)
- 🌍 All Whisper languages for final transcription

## License

MIT
