# Showtime — AI Context

## What is Showtime?

Showtime turns a rough screen recording + voiceover audio into a polished, synced demo video. Uses vision AI (Llama 4 Scout) to see what's on screen and match narration to the right visual moments.

## Architecture

### MVC (`app/`)

| Layer | Path | Role |
|-------|------|------|
| Model | `app/models/domain.py` | VideoSegment, MappingEntry (freeze flag), Timeline, TimelineClip (is_gap) |
| Model | `app/models/schemas.py` | PipelineInput/Result, API request/response schemas |
| View | `app/api/v1/` | FastAPI routes: upload, process, status, download |
| Controller | `app/services/` | Pipeline steps + orchestration |
| Core | `app/core/` | Config (pydantic-settings + .env) + exceptions |

### Pipeline

```
Video → Scene Detector (OpenCV + auto-split + merge) → Keyframes
                                                           ↓
Audio → Whisper (Groq) → Sentences          AI Mapper (Vision → Text → Chronological)
                                                ↓                    ↓
                                          Refinement Loop (max 2)   ↓
                                                ↓                    ↓
                                            Timeline (gaps + sub-segment split)
                                                ↓
                                         Renderer (play/freeze/gap clips + concat)
                                                ↓
                                            output.mp4
```

### Scene Detector (`app/services/scene_detector.py`)

1. Sample frames at `SAMPLE_FPS` (default 2/sec)
2. Compute pixel diffs → scene boundaries where diff > `SCENE_THRESHOLD`
3. **Auto-split**: segments > `MAX_SEGMENT_DURATION` (10s) re-scanned with lower threshold (20.0)
4. **Merge**: segments < `MIN_SEGMENT_DURATION` (1.5s) absorbed into neighbors (skip for videos <10s)
5. **Idle detection**: consecutive low-diff frames → `is_idle=True` (filtered out in pipeline)
6. Extract keyframe (middle frame) per segment as PNG

### AI Mapper (`app/services/ai_mapper.py`)

**3-tier strategy + refinement loop:**

1. **Vision (Llama 4 Scout)** — sends base64 keyframe images via Groq chat API. Model SEES screens, decides per-sentence:
   - `segment_id`: which visual segment matches
   - `freeze: bool`: PLAY (action) or FREEZE (hold still for reading). Max 3 freezes (capped by `_cap_freeze_count`)

2. **Text (Llama 3.3 70B)** — fallback, sends OCR descriptions

3. **Scene-aware chronological** — maps by time scaling: `video_time = audio_time * (total_video / total_audio)`

4. **Refinement loop** (max 2 passes):
   - `_analyze_mapping()` builds pacing analysis with WARNING flags (speed >2.0x, <0.5x, short clip <1.5s, long freeze >8s)
   - AI reviews its own mapping + warnings, responds `{"action": "keep"}` or `{"action": "refine", "mappings": [...]}`
   - Only accepted if warnings decrease
   - All passes enforce `_is_chronological()` — segment starts must be non-decreasing

### Timeline (`app/services/timeline.py`)

- `_compute_video_positions()` distributes video across ALL events (sentences + gaps)
- **Sub-segment splitting**: shared segments split proportionally by audio duration
- **Gap clips**: silent pauses > 0.15s → video continues playing with silence

### Renderer (`app/services/renderer.py`)

Three clip types:
- **Content** (`_render_content_clip`): speed-adjusted video + trimmed audio. Speed clamped 0.5-2.5x (auto-freezes if >2.5x)
- **Freeze** (`_render_freeze_clip`): single keyframe looped + voiceover audio
- **Gap** (`_render_gap_clip`): video at speed + generated silence (`anullsrc`), also respects speed clamps

All clips: same fps, `-ar 44100 -ac 2`. Concatenated via concat demuxer. Output validated with ffprobe.

## Tech Stack

| Component | Tool |
|-----------|------|
| Python 3.12, uv | Language + package manager |
| FastAPI + Uvicorn | Web framework |
| Typer + Rich | CLI |
| OpenCV | Scene detection + keyframe extraction |
| Tesseract | OCR + structural visual analysis |
| Groq Whisper | Audio transcription (word timestamps) |
| Groq Llama 4 Scout | Vision-based sentence→segment mapping |
| Groq Llama 3.3 70B | Text-only mapping fallback |
| FFmpeg | Video rendering (play/freeze/gap clips) |
| Pydantic v2 | Data validation |
| pytest (159 tests) | Testing |

## Import Patterns

```python
from app.core.config import Settings
from app.core.exceptions import ShowtimeError, RenderError
from app.models.domain import VideoSegment, MappingEntry, Timeline, TimelineClip
from app.models.schemas import PipelineInput, PipelineResult, JobStatus
from app.services.pipeline import run_pipeline
from app.services.ai_mapper import map_sentences_to_segments
```

## Key Config (`.env`)

```bash
SHOWTIME_GROQ_API_KEY=...                                        # required
SHOWTIME_GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
SHOWTIME_GROQ_MODEL=llama-3.3-70b-versatile
SHOWTIME_GROQ_WHISPER_MODEL=whisper-large-v3-turbo
SHOWTIME_SCENE_THRESHOLD=30.0
SHOWTIME_SCENE_REFINE_THRESHOLD=20.0
SHOWTIME_MAX_SEGMENT_DURATION=10.0
SHOWTIME_MIN_SEGMENT_DURATION=1.5
SHOWTIME_MAX_PLAYBACK_SPEED=2.5
SHOWTIME_MIN_PLAYBACK_SPEED=0.5
```

## What's Not Built Yet

- **Frontend** — Next.js app (ready to start)
- **Celery workers** — async processing (currently threads)
- **Persistent job store** — Redis/DB (currently in-memory dict)
- **Phase 2** — auto script generation, TTS voiceover
