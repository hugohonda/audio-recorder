# Audio Recorder

Capture macOS system audio (Zoom calls, YouTube, Spotify, etc.) and microphone using Apple's ScreenCaptureKit framework. Live transcription with Moonshine, high-quality final transcription with MLX Whisper.

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
# Record until Ctrl+C (saves to recordings/YYYY-MM-DD_HH-MM-SS.mp3)
uv run audio-recorder record

# Record for specific duration
uv run audio-recorder record -d 60

# Custom output filename
uv run audio-recorder record -o meeting.mp3

# Save as WAV instead of MP3
uv run audio-recorder record -o meeting.wav

# System audio only (no microphone)
uv run audio-recorder record --no-mic

# Live transcription while recording (Moonshine)
uv run audio-recorder record --live

# Skip Whisper transcription after recording
uv run audio-recorder record --no-final

# Use a specific Whisper model for final transcription
uv run audio-recorder record -m tiny

# Live transcription only, no final Whisper pass
uv run audio-recorder record --live --no-final -d 30
```

### Transcribe

Re-transcribe an existing audio file with Whisper:

```bash
uv run audio-recorder transcribe recording.mp3
uv run audio-recorder transcribe recording.mp3 -m large
```

### List Models

```bash
uv run audio-recorder models
```

**Default output:**
- `recordings/YYYY-MM-DD_HH-MM-SS.mp3` (system audio)
- `recordings/YYYY-MM-DD_HH-MM-SS_mic.mp3` (microphone, when enabled)

## First Run

On first run, macOS will prompt for **Screen Recording** permission:

`System Settings > Privacy & Security > Screen Recording`

## How It Works

1. **ScreenCaptureKit** captures system audio at 16kHz mono
2. **Microphone** captured at 24kHz, resampled to 16kHz for saving
3. **Saved as separate files** (system + mic) in MP3 or WAV
4. **Live transcription** (optional): Moonshine runs on the audio buffer every 3s, showing new words incrementally
5. **Final transcription** (default): MLX Whisper runs on the saved file for high-quality output

### Output Format (Optimized for Transcription)

| Parameter | Value | Why |
|-----------|-------|-----|
| Format | MP3 | ~10x smaller than WAV |
| Sample Rate | 16 kHz | Whisper/Moonshine native rate |
| Channels | Mono | Stereo doubles size with no benefit |
| Bitrate | 64 kbps | Plenty for speech clarity |

**File size:** ~480 KB per minute (~29 MB per hour)

## Project Structure

```
audio-recorder/
├── recordings/          # Output folder (gitignored)
├── src/audio_recorder/
│   ├── __init__.py      # Package exports
│   ├── cli.py           # Click CLI (record, transcribe, models)
│   ├── capture.py       # ScreenCaptureKit audio capture
│   ├── audio.py         # Audio utilities (conversion, resampling, buffer)
│   └── moonshine_transcriber.py  # Live transcription (Moonshine, sliding window)
├── tests/
├── pyproject.toml
└── AGENTS.md
```

## Dependencies

- `pyobjc-framework-ScreenCaptureKit` - macOS audio capture
- `pyobjc-framework-CoreMedia` - Audio buffer handling
- `lameenc` - MP3 encoding (LAME)
- `click` - CLI framework
- `mlx-whisper` - Final transcription (MLX Whisper)
- `useful-moonshine-onnx` - Live transcription (Moonshine)

## License

MIT
