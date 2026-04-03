.PHONY: record record-mic record-pt record-pt-mic record-live record-live-mic record-pt-live record-pt-live-mic transcribe summarize models lint format

# English recording
record:
	uv run audio-recorder record

record-mic:
	uv run audio-recorder record --mic-only

record-live:
	uv run audio-recorder record --live

record-live-mic:
	uv run audio-recorder record --live --mic-only

# Portuguese recording
record-pt:
	uv run audio-recorder record -l pt-br

record-pt-mic:
	uv run audio-recorder record -l pt-br --mic-only

record-pt-live:
	uv run audio-recorder record --live -l pt-br

record-pt-live-mic:
	uv run audio-recorder record --live -l pt-br --mic-only

# Utilities
transcribe:
	@echo "Usage: make transcribe FILE=path/to/audio.mp3"
	uv run audio-recorder transcribe $(FILE)

summarize:
	@echo "Usage: make summarize FILE=path/to/transcript.txt"
	uv run audio-recorder summarize $(FILE)

models:
	uv run audio-recorder models

# Dev
lint:
	uv run ruff check . --fix && uv run ruff format .

test:
	uv run pytest -q
