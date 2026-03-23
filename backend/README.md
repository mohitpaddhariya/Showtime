# Showtime Backend

Turn a rough screen recording + voiceover audio into a polished, synced demo video — automatically.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- `brew install ffmpeg tesseract`
- Free [Groq API key](https://console.groq.com/keys)

### Install

```bash
cd backend
cp .env.example .env    # add SHOWTIME_GROQ_API_KEY
uv venv && uv pip install -e ".[dev]"
```

### Usage

```bash
# CLI
uv run showtime recording.mp4 voiceover.mp3 -o output.mp4

# E2E test with timing breakdown
uv run python scripts/test_real_video.py recording.mp4 voiceover.mp3 -o output.mp4

# API server
uv run uvicorn app.main:app --reload
```

### API Endpoints

```bash
curl -X POST localhost:8000/api/v1/upload -F video=@rec.mp4 -F audio=@vo.mp3
curl -X POST localhost:8000/api/v1/process/{job_id}
curl localhost:8000/api/v1/status/{job_id}
curl -OJ localhost:8000/api/v1/download/{job_id}
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/upload` | Upload video + audio → job_id |
| `POST` | `/api/v1/process/{job_id}` | Start processing (background) |
| `GET` | `/api/v1/status/{job_id}` | Progress 0-100% |
| `GET` | `/api/v1/download/{job_id}` | Download rendered .mp4 |
| `GET` | `/api/v1/result/{job_id}` | Result metadata (duration, clips, etc.) |

## Architecture (MVC)

```
app/
├── core/       # Config + exceptions
├── models/     # M — domain models + API schemas
├── services/   # C — pipeline steps + orchestration
└── api/        # V — FastAPI routes
```

### Pipeline

```
Screen Recording → Scene Detector → Keyframes ─┐
                   (auto-split + merge)         │
                                                ├→ AI Mapper (Vision) → Timeline → Renderer → .mp4
Voiceover Audio → Whisper (Groq) → Sentences ──┘   + Refinement Loop
```

**6 steps:**

1. **Scene Detector** — OpenCV frame diffs, auto-splits long segments (>10s) with lower threshold, merges tiny segments (<1.5s), removes idle segments
2. **Frame Captioner** — Tesseract OCR + structural analysis (colors, layout, UI density)
3. **Audio Analyzer** — Groq Whisper transcription with word-level timestamps → sentence segmentation
4. **AI Mapper** — 3-tier + refinement:
   - **Llama 4 Scout (vision)** — sees keyframe images, matches narration to screen content, decides PLAY vs FREEZE per sentence
   - **Llama 3.3 70B (text)** — fallback using OCR descriptions
   - **Scene-aware chronological** — pure math fallback
   - **Refinement loop** (max 2 passes) — AI reviews its own mapping, sees pacing warnings, adjusts
   - All strategies enforce chronological order (no backward jumps)
5. **Timeline** — Sub-segment splitting, gap clips with silence for natural pauses
6. **Renderer** — Per-clip video+audio sync (play/freeze/gap), fps normalization, ffprobe validation

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SHOWTIME_GROQ_API_KEY` | — | Required |
| `SHOWTIME_GROQ_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Vision mapper |
| `SHOWTIME_GROQ_MODEL` | `llama-3.3-70b-versatile` | Text fallback |
| `SHOWTIME_GROQ_WHISPER_MODEL` | `whisper-large-v3-turbo` | Transcription |
| `SHOWTIME_SCENE_THRESHOLD` | `30.0` | Scene change sensitivity |
| `SHOWTIME_SCENE_REFINE_THRESHOLD` | `20.0` | Lower threshold for splitting long segments |
| `SHOWTIME_MAX_SEGMENT_DURATION` | `10.0` | Auto-split segments longer than this |
| `SHOWTIME_MIN_SEGMENT_DURATION` | `1.5` | Merge segments shorter than this |
| `SHOWTIME_OUTPUT_PRESET` | `medium` | FFmpeg preset |
| `SHOWTIME_CRF` | `23` | Video quality (18-28) |
| `SHOWTIME_CROSSFADE_DURATION` | `0.3` | Transition duration (0 = hard cut) |

### Supported Formats

| Input video | `.mp4` `.mov` `.avi` `.mkv` `.webm` `.flv` `.wmv` |
|-------------|-----------------------------------------------------|
| Input audio | `.mp3` `.wav` `.m4a` `.aac` `.ogg` `.flac` `.wma` |
| Output | `.mp4` (H.264 + AAC) |

## Testing

```bash
uv run pytest                       # all 159 tests
uv run pytest -m "not integration"  # unit tests (fast)
uv run pytest tests/test_api.py -v  # API tests
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI
│   ├── cli.py               # Typer CLI
│   ├── core/config.py       # Settings (.env)
│   ├── core/exceptions.py   # Error hierarchy
│   ├── models/domain.py     # VideoSegment, MappingEntry (freeze), Timeline
│   ├── models/schemas.py    # PipelineInput/Result, API schemas
│   ├── services/
│   │   ├── pipeline.py      # Orchestration + idle removal
│   │   ├── scene_detector.py # OpenCV + auto-split + merge
│   │   ├── frame_captioner.py # OCR + structural analysis
│   │   ├── audio_analyzer.py  # Groq/local Whisper
│   │   ├── ai_mapper.py      # Vision + text + fallback + refinement
│   │   ├── timeline.py       # EDL + gaps + sub-splitting
│   │   └── renderer.py       # FFmpeg play/freeze/gap clips
│   └── api/v1/              # upload, process, status, download
├── tests/                   # 159 tests
├── scripts/test_real_video.py
└── .env.example
```
