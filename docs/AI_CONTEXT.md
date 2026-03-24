# Showtime — AI Context

## What is Showtime?

Showtime turns a rough screen recording + voiceover audio into a polished, synced demo video. Uses vision AI (Llama 4 Scout) to see what's on screen and match narration to the right visual moments.

## Architecture

### MVC (`app/`)

| Layer | Path | Role |
|-------|------|------|
| Model | `app/models/domain.py` | VideoSegment (semantic_tag), MappingEntry (freeze, confidence, reasoning), Timeline, TimelineClip (is_gap) |
| Model | `app/models/schemas.py` | PipelineInput/Result, API request/response schemas |
| View | `app/api/v1/` | FastAPI routes: upload, process, status, download |
| Controller | `app/services/` | Pipeline steps + orchestration |
| Helper | `app/services/vision_utils.py` | Shared keyframe resize/base64/batching for Groq vision calls |
| Core | `app/core/` | Config (pydantic-settings + .env) + exceptions |

### Pipeline (v2)

```
Video --> Scene Detector (OpenCV + auto-split + merge) --> Keyframes
               |                                              |
               +---> AI Verification (1 Groq vision call) <---+
                     Confirm/merge/split + semantic tags
                                    |
                                    v
Audio --> Whisper (Groq) --> Sentences     AI Mapper (1 vision call, ALL keyframes batched)
                                              |         Built-in self-critique (pacing_score)
                                              v
                                    Optional Refinement (1 text call if score < 8.5)
                                              |
                                              v
                                    Timeline (proportional split + auto-freeze + gaps)
                                              |
                                              v
                                    Renderer (content/freeze+KenBurns/gap-freeze clips)
                                              |
                                              v
                                          output.mp4

Groq call budget: 1 Whisper + 1 AI Verify + 1 AI Map + 0-1 Refine = 3-4 total
```

### Scene Detector (`app/services/scene_detector.py`)

1. Sample frames at `SAMPLE_FPS` (default 2/sec)
2. Compute pixel diffs + histogram comparison --> scene boundaries where diff > `SCENE_THRESHOLD`
3. **Auto-split**: segments > `MAX_SEGMENT_DURATION` (8s) re-scanned with lower threshold (20.0)
4. **Merge**: segments < `MIN_SEGMENT_DURATION` (1.5s) absorbed into neighbors (skip for videos <10s)
5. **Idle detection**: consecutive low-diff frames --> `is_idle=True` (filtered out in pipeline)
6. Extract keyframe (middle frame) per segment as PNG
7. **[v2] AI Verification**: batch ALL keyframes in ONE Llama 4 Scout call --> confirm/merge/split segments semantically + assign `semantic_tag` per segment (e.g. "landing_page", "code_editor")

### AI Mapper (`app/services/ai_mapper.py`)

**v2: Single vision call with built-in self-critique (replaces old 3-tier + 2-pass refinement):**

1. **Vision Mapping (Llama 4 Scout)** -- ONE call with ALL keyframes batched + full transcript. Model SEES screens and matches narration to visual content. Returns per-sentence:
   - `segment_id`: which visual segment matches
   - `freeze: bool`: PLAY (action) or FREEZE (hold still for reading). Max 3 freezes
   - `confidence: float`: 0.0-1.0 mapping confidence
   - `reasoning: str`: brief explanation
   - `pacing_score: float`: 0-10 self-assessed pacing quality

2. **Optional Refinement** (max 1 text call) -- only triggered if `pacing_score < PACING_THRESHOLD` (default 8.5). Uses Llama 3.3 70B to review pacing analysis and adjust.

3. **Fallbacks** -- text mapping (Groq/Ollama) if no keyframes; chronological mapping if all AI fails.

4. All strategies enforce `_is_chronological()` -- segment starts must be non-decreasing.

### Timeline (`app/services/timeline.py`)

- `_compute_video_positions()` distributes video across ALL events (sentences + gaps)
- **Proportional sub-segment splitting**: shared segments split proportionally by audio duration, with normalization guard to prevent overflow
- **Audio > video guard**: when minimum-duration enforcement would overflow a segment, allocations are scaled down proportionally
- **Auto-freeze guard**: zero-duration clips (segment exhausted) auto-converted to freeze frames
- **Gap clips**: silent pauses > 0.15s --> gap clip inserted

### Renderer (`app/services/renderer.py`)

Four clip rendering modes:
- **Content** (`_render_content_clip`): speed-adjusted video + trimmed audio. Speed clamped 0.5-2.5x (auto-freezes if >2.5x)
- **Freeze** (`_render_freeze_clip`): Ken Burns subtle zoom (1.0x-->1.03x) on keyframe + voiceover audio. Prevents dead-screen effect.
- **Gap** (`_render_gap_clip`): when enough video, plays at speed + silence. When video too short (speed < 0.5x), holds a clean still frame + silence instead of laggy slow-motion.
- **Gap-Freeze**: automatic fallback for gap clips where video is too short for smooth playback.

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
| Groq Llama 4 Scout | Vision: scene verification + sentence-to-segment mapping |
| Groq Llama 3.3 70B | Text-only mapping fallback + refinement |
| FFmpeg | Video rendering (content/freeze/gap clips + Ken Burns) |
| Pydantic v2 | Data validation |
| pytest (165 tests) | Testing |

## Import Patterns

```python
from app.core.config import Settings
from app.core.exceptions import ShowtimeError, RenderError
from app.models.domain import VideoSegment, MappingEntry, Timeline, TimelineClip
from app.models.schemas import PipelineInput, PipelineResult, JobStatus
from app.services.pipeline import run_pipeline
from app.services.ai_mapper import map_sentences_to_segments
from app.services.vision_utils import encode_keyframe_b64, build_image_content_blocks
```

## Key Config (`.env`)

```bash
SHOWTIME_GROQ_API_KEY=...                                        # required
SHOWTIME_GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
SHOWTIME_GROQ_MODEL=llama-3.3-70b-versatile
SHOWTIME_GROQ_WHISPER_MODEL=whisper-large-v3-turbo
SHOWTIME_SCENE_THRESHOLD=30.0
SHOWTIME_SCENE_REFINE_THRESHOLD=20.0
SHOWTIME_MAX_SEGMENT_DURATION=8.0
SHOWTIME_MIN_SEGMENT_DURATION=1.5
SHOWTIME_MAX_PLAYBACK_SPEED=2.5
SHOWTIME_MIN_PLAYBACK_SPEED=0.5
# v2 additions
SHOWTIME_AI_VERIFY_SCENES=true          # AI scene verification (1 Groq call)
SHOWTIME_MAX_FREEZE_COUNT=3             # max freeze frames per video
SHOWTIME_PACING_THRESHOLD=8.5           # self-critique score triggering refinement
SHOWTIME_KEN_BURNS_ON_FREEZE=true       # subtle zoom on freeze clips
SHOWTIME_KEN_BURNS_ZOOM=1.03            # max zoom factor (3%)
```

## What's Not Built Yet

- **Frontend** -- Next.js app (ready to start)
- **Celery workers** -- async processing (currently threads)
- **Persistent job store** -- Redis/DB (currently in-memory dict)
- **Phase 2** -- auto script generation, TTS voiceover
