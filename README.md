# Showtime

Turn rough screen recordings into polished, synced demo videos — automatically.

Record your screen. Record your voiceover. Showtime handles the edit.

---

## What it does

Showtime takes a screen recording and a voiceover, then:

1. Detects scene changes in the recording
2. Transcribes the voiceover with word-level timestamps
3. Maps each sentence to the right moment on screen using AI vision
4. Renders a final video with smooth transitions and adaptive playback speed

---

## Setup

**Requirements:** Python 3.12+, `ffmpeg`, `tesseract`, a [Groq API key](https://console.groq.com)

```bash
# macOS
brew install ffmpeg tesseract

# Clone and install
git clone https://github.com/your-username/showtime.git
cd showtime/backend
cp .env.example .env          # add your SHOWTIME_GROQ_API_KEY
uv pip install -e ".[dev]"
```

---

## Usage

**CLI**
```bash
uv run showtime recording.mp4 voiceover.mp3 -o output.mp4
```

**API server**
```bash
uv run uvicorn app.main:app --reload
# Docs → http://localhost:8000/docs
```

**API workflow**
```bash
# 1. Upload
curl -X POST http://localhost:8000/api/v1/upload \
  -F "video=@recording.mp4" -F "audio=@voiceover.mp3"
# → { "job_id": "abc123" }

# 2. Process
curl -X POST http://localhost:8000/api/v1/process/abc123

# 3. Poll status
curl http://localhost:8000/api/v1/status/abc123

# 4. Download
curl -O http://localhost:8000/api/v1/download/abc123
```

---

## Configuration

Set via environment variables in `.env`. Only one is required:

| Variable | Description |
|----------|-------------|
| `SHOWTIME_GROQ_API_KEY` | Groq API key (required) |

See `.env.example` for all options (models, scene detection sensitivity, playback speed limits, video quality, etc).

---

## Development

```bash
uv run pytest                        # all tests
uv run pytest -m "not integration"   # unit tests only
uv run pytest --cov=app              # with coverage
```

---

## Stack

FastAPI · Groq (Whisper + Llama 4 Scout) · OpenCV · Tesseract · FFmpeg