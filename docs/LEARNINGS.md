# Showtime — Complete Technical Learnings

Every concept, technique, algorithm, and design pattern used in this codebase, explained for deep learning.

---

## Table of Contents

1. [Architecture & Design Patterns](#1-architecture--design-patterns)
2. [Computer Vision & Image Processing](#2-computer-vision--image-processing)
3. [AI & LLM Integration](#3-ai--llm-integration)
4. [Audio Processing & NLP](#4-audio-processing--nlp)
5. [Video Rendering & FFmpeg](#5-video-rendering--ffmpeg)
6. [Timeline & Sync Algorithms](#6-timeline--sync-algorithms)
7. [Backend & API Design](#7-backend--api-design)
8. [Testing Strategies](#8-testing-strategies)
9. [Data Modeling & Validation](#9-data-modeling--validation)
10. [Configuration & DevOps](#10-configuration--devops)
11. [Non-Obvious Tricks & Optimizations](#11-non-obvious-tricks--optimizations)

---

## 1. Architecture & Design Patterns

### 1.1 Pipeline Pattern (Sequential Processing Chain)

**What it is:** A series of processing stages where each stage's output feeds the next. The entire app is a 6-step pipeline where raw inputs flow through detection → captioning → transcription → mapping → timeline → rendering.

**How it works in the code:**

```python
# app/services/pipeline.py
def run_pipeline(pipeline_input, settings, on_progress):
    segments  = detect_scenes(...)        # Step 1
    captioned = caption_segments(segments) # Step 2
    sentences = transcribe_audio(...)      # Step 3
    mappings  = map_sentences_to_segments(captioned, sentences, settings)  # Step 4
    timeline  = assemble_timeline(mappings, captioned, sentences, ...)     # Step 5
    render(timeline, output_path, settings)                                # Step 6
```

**Why it works:** Each step is a pure function (input → output) with no shared mutable state. This makes testing trivial — you can test each step independently with mock inputs.

**Design insight:** The `on_progress` callback allows the CLI/API to report progress without coupling the pipeline to any UI framework. This is the **Observer Pattern** applied to pipeline orchestration.

**How to learn this:**
- *Design Patterns* by Gang of Four — Pipeline/Chain of Responsibility
- Martin Fowler's *Pipes and Filters* enterprise pattern
- Study Unix philosophy: `cat file | grep pattern | sort | uniq`

---

### 1.2 Strategy Pattern (Multi-Tier AI Fallback)

**What it is:** Multiple interchangeable algorithms that solve the same problem, tried in priority order with automatic fallback.

**How it works:**

```python
# app/services/ai_mapper.py
def map_sentences_to_segments(segments, sentences, settings):
    # Strategy 1: Vision (Llama 4 Scout — sees actual keyframes)
    if settings.groq_api_key and has_keyframes:
        result, pacing_score = _vision_mapping(...)
        if _is_chronological(result, segments_by_id):
            return result

    # Strategy 2: Text (Llama 3.3 70B — reads OCR descriptions)
    raw = _call_groq_text(segments, sentences, settings)
    result = _parse_response(raw, segments, sentences)
    if _is_chronological(result, segments_by_id):
        return result

    # Strategy 3: Chronological (zero API calls — pure math)
    return _scene_aware_mapping(segments, sentences)
```

**Why 3 tiers?** Vision can fail (no keyframes, API down). Text can fail (bad OCR, rate limits). The chronological fallback uses zero API calls — it **always works** by mapping sentence midpoints to proportional video positions.

**How to learn this:**
- *Head First Design Patterns* — Strategy Pattern chapter
- Study how browsers implement font fallback: `font-family: Inter, Helvetica, Arial, sans-serif`

---

### 1.3 MVC Architecture (Model-View-Controller)

**What it is:** Separation of data (Models), presentation (API routes/CLI), and business logic (Services).

```
app/models/domain.py   → Data structures (VideoSegment, Timeline, etc.)
app/models/schemas.py  → API request/response schemas
app/api/v1/            → REST endpoints (Views)
app/services/          → Pipeline steps (Controllers)
app/core/              → Config + exceptions (Infrastructure)
```

**Key insight:** Models use Pydantic `BaseModel` with `@property` decorators for computed fields. This keeps derived values (like `duration`) always in sync:

```python
class VideoSegment(BaseModel):
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start  # Always correct, never stale
```

---

### 1.4 Custom Exception Hierarchy

**What it is:** A tree of domain-specific exceptions rooted at a single base class.

```python
class ShowtimeError(Exception): ...      # Base — catch-all
class SceneDetectionError(ShowtimeError): ...
class CaptionError(ShowtimeError): ...
class TranscriptionError(ShowtimeError): ...
class MappingError(ShowtimeError): ...
class TimelineError(ShowtimeError): ...
class RenderError(ShowtimeError): ...
```

**Why:** Each pipeline step raises its own exception type. The CLI catches `ShowtimeError` at the top level — one `except` handles all pipeline failures with the right error message. Callers can also catch specific types for fine-grained handling.

**How to learn this:**
- Python docs: [User-defined Exceptions](https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions)
- Study how Django, FastAPI, and SQLAlchemy define exception hierarchies

---

## 2. Computer Vision & Image Processing

### 2.1 Frame Differencing for Scene Detection

**What it is:** Comparing consecutive video frames pixel-by-pixel to find where visual content changes significantly (scene boundaries).

**Algorithm (dual-method):**

```python
# Method 1: Mean Absolute Pixel Difference
diff = cv2.absdiff(frames[i][1], frames[i - 1][1])  # Per-pixel abs diff
mean_diff = float(np.mean(diff))                      # Average across all pixels

# Method 2: Histogram Correlation (catches color/layout changes)
if mean_diff > settings.scene_threshold * 0.5:        # Gate: only if some change
    hist_prev = cv2.calcHist([frames[i-1][1]], [0], None, [64], [0, 256])
    hist_curr = cv2.calcHist([frames[i][1]], [0], None, [64], [0, 256])
    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)
    hist_corr = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
    hist_change = hist_corr < 0.85  # Low correlation = significant change
```

**Why two methods?**
- Pixel diff catches structural changes (new UI elements, text appearing)
- Histogram correlation catches color/layout shifts that pixel diff misses (dark mode toggle, page navigation where overall brightness stays similar but content rearranges)
- The 50% gating threshold prevents histogram from overriding when there's almost no pixel change (reduces false positives)

**How to learn this:**
- OpenCV docs: `cv2.absdiff`, `cv2.calcHist`, `cv2.compareHist`
- PyImageSearch tutorials on video analysis
- Paper: "A Survey on Video Scene Detection" for academic depth

---

### 2.2 Adaptive Frame Sampling

**What it is:** Instead of analyzing every frame (30+ fps × minutes = thousands), sample at a lower rate.

```python
sample_interval = max(1, int(fps / settings.sample_fps))  # e.g., 30fps / 2 = every 15th frame
for frame_idx in range(0, total_frames, sample_interval):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale = 1/3 memory
```

**Memory optimization:** Converting to grayscale uses 1 channel instead of 3, cutting memory by ~66%.

**Tradeoff:** `sample_fps=2` (default) catches transitions ≥0.5s apart. `sample_fps=4` catches ≥0.25s transitions but uses 2× more memory.

---

### 2.3 Multi-Pass Segment Refinement

**What it is:** Three-pass approach: detect → split long → merge short.

```
Pass 1: Primary detection (threshold=30.0)
    → Might produce segments like: [0-15s, 15s-40s]  (40s segment is too long)

Pass 2: Auto-split (threshold=20.0 for segments > 8s)
    → Re-scans the 25s segment with lower threshold
    → [0-15s, 15s-22s, 22s-35s, 35s-40s]

Pass 3: Merge short (< 1.5s absorbed into neighbors)
    → Removes noise from cursor blinks, brief UI flickers
```

**Key insight:** Pass 2 uses a *lower* threshold specifically for long segments. This is because the primary threshold might be too aggressive (misses slow scrolling), but lowering it globally would create too many tiny segments. The two-threshold system balances sensitivity.

---

### 2.4 Idle Detection

**What it is:** Finding segments where nothing happens on screen (loading spinners, blank screens).

```python
def _check_idle(b_idx, boundaries, boundary_pos, frames, settings):
    # All consecutive diffs must be below threshold
    for j in range(start_idx + 1, end_idx):
        diff = cv2.absdiff(frames[j][1], frames[j - 1][1])
        if float(np.mean(diff)) > settings.idle_threshold:
            return False  # Something changed — not idle
    return True  # Nothing changed for the entire segment
```

**Why remove idle?** The product pitch is "cuts out boring parts." Idle segments add nothing to a demo video.

---

### 2.5 Canny Edge Detection for UI Density

**What it is:** Using edge detection to measure how "busy" a screen is.

```python
edges = cv2.Canny(gray, 50, 150)
edge_density = np.count_nonzero(edges) / (h * w)

if edge_density > 0.15:   features.append("complex/busy layout")
elif edge_density > 0.05: features.append("moderate layout")
else:                      features.append("simple/minimal layout")
```

**Why:** Gives the AI mapper context about the visual complexity of each screen. A code editor (high density) looks different from a login page (low density).

**How to learn this:**
- OpenCV docs: `cv2.Canny` (hysteris thresholding, Sobel gradients)
- "Learning OpenCV 4" by Kaehler & Bradski

---

### 2.6 Contour Detection for UI Elements

**What it is:** Finding rectangular shapes (buttons, cards, form fields) in screen captures.

```python
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, cw, ch = cv2.boundingRect(cnt)
    area_ratio = (cw * ch) / (w * h)
    if 0.01 < area_ratio < 0.5 and 0.2 < cw / max(ch, 1) < 5:
        large_rects += 1  # Reasonable aspect ratio, not too small or large
```

**Filters applied:**
- `area_ratio > 0.01` — ignores tiny artifacts (noise, dots)
- `area_ratio < 0.5` — ignores rectangles that are half the screen (background)
- Aspect ratio `0.2 < w/h < 5` — filters out extreme shapes (thin lines)

---

### 2.7 Dominant Color Analysis (Center-Weighted)

```python
# Sample center region to avoid chrome/toolbars
margin_h, margin_w = h // 10, w // 10
center = img[margin_h:h - margin_h, margin_w:w - margin_w]
mean_color = center.mean(axis=(0, 1))  # BGR average
```

**Trick:** Only analyzing the center 80% of the image avoids picking up browser chrome, taskbar, and title bar colors that aren't part of the actual content.

---

## 3. AI & LLM Integration

### 3.1 Batched Multimodal Vision Calls

**What it is:** Sending ALL keyframe images in a single API call instead of one per segment.

```python
def build_image_content_blocks(segments, max_dim=1024):
    content = []
    for seg in segments:
        content.append({"type": "text", "text": f"--- Segment {seg.segment_id} ---"})
        b64 = encode_keyframe_b64(seg.keyframe_path, max_dim)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return content
```

**Why batch?** Groq free tier has 15 req/min for vision. Sending 8 images separately = 8 calls = rate limited. Batching = 1 call. This brings the entire mapper from 5 calls (v1) down to 1-2 calls (v2).

**Image limit workaround:** Groq's Llama 4 Scout limits 5 images per request. When >5 segments exist, the code picks 5 evenly-spaced images and includes the rest as text descriptions:

```python
if len(segments) > _MAX_VISION_IMAGES:
    step = len(segments) / _MAX_VISION_IMAGES
    image_indices = {int(i * step) for i in range(_MAX_VISION_IMAGES)}
    for idx, seg in enumerate(segments):
        if idx in image_indices:
            content.extend(build_image_content_blocks([seg]))  # Image
        else:
            content.append({"type": "text", "text": f"[Text only] {seg.description}"})
```

---

### 3.2 Image Payload Optimization

```python
def resize_keyframe(keyframe_path, max_dim=1024):
    img = cv2.imread(str(keyframe_path))
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()
```

**Math:** A 1920×1080 screenshot is ~400KB raw, ~550KB base64. Resizing to 1024×576 brings it to ~150KB base64. 8 images: 4.4MB → 1.2MB (well under Groq's 4MB limit).

**`INTER_AREA` interpolation:** Best for downscaling — it averages pixel neighborhoods rather than picking nearest neighbors, avoiding aliasing artifacts on text.

---

### 3.3 Self-Critique Pattern (AI Reviews Its Own Work)

**What it is:** The vision model rates its own mapping quality with a `pacing_score` (0-10). If the score is below 8.5, a cheaper text model reviews and fixes it.

```python
result, pacing_score = _vision_mapping(segments, sentences, settings)
if pacing_score < settings.pacing_threshold:  # 8.5 default
    refined = _refine_once(result, segments, sentences, ...)
    if refined is not None:
        result = refined
```

**Why self-critique?** The vision model is expensive (images). The text model is cheap (just numbers). By having the vision model flag its own uncertainty, we only trigger the refinement call when needed — saving 50% of API calls on average.

**Guard rail — warning count regression:** Refinement is only accepted if it *actually reduces* warning count:

```python
old_warnings = analysis.count("WARNING")
new_warnings = new_analysis.count("WARNING")
if new_warnings < old_warnings:
    return refined  # Accept
return None  # Reject — refinement made things worse
```

---

### 3.4 Structured JSON Mode

```python
response = client.chat.completions.create(
    model=settings.groq_vision_model,
    messages=[...],
    temperature=0.1,                          # Low randomness
    response_format={"type": "json_object"},  # Forces valid JSON output
)
```

**Why `temperature=0.1`?** Mapping is deterministic work (sentence X → segment Y). High temperature would randomize the mapping, producing inconsistent results.

**Why JSON mode?** Without it, LLMs wrap JSON in markdown code blocks, add explanatory prose, or produce malformed output. `json_object` mode guarantees parseable output.

---

### 3.5 Chronological Enforcement

```python
def _is_chronological(mappings, segments_by_id):
    prev_start = -1.0
    for m in mappings:
        seg = segments_by_id.get(m.segment_id)
        if seg.start < prev_start:
            return False  # Backward jump detected!
        prev_start = seg.start
    return True
```

**Why:** LLMs sometimes produce non-chronological mappings (sentence 5 → segment 3, sentence 6 → segment 1). This creates jarring backward video jumps. The constraint ensures the viewer only moves forward through the recording.

---

### 3.6 Freeze Cap (Anti-Slideshow)

```python
def _cap_freeze_count(entries, max_freeze=3):
    freeze_entries = [(i, e) for i, e in enumerate(entries) if e.freeze]
    keep_indices = {i for i, _ in freeze_entries[-max_freeze:]}
    # Only keep the LAST N freezes (later = more likely to show results)
```

**Why keep the last N?** Demo videos typically end with results/summaries — these are the most valuable moments to freeze on. Early freezes are often intro screens that don't need to be held.

---

## 4. Audio Processing & NLP

### 4.1 Dual-Provider Transcription

**Cloud (Groq Whisper):**
```python
response = client.audio.transcriptions.create(
    file=(audio_path.name, audio_file),
    model="whisper-large-v3-turbo",
    response_format="verbose_json",
    timestamp_granularities=["word", "segment"],
)
```

**Local (OpenAI Whisper):**
```python
model = whisper.load_model(settings.whisper_model)
result = model.transcribe(str(audio_path), word_timestamps=True)
```

**Why both?** Groq is 10-50× faster but requires internet + API key. Local works offline — critical for development and air-gapped environments.

---

### 4.2 Word → Sentence Grouping

```python
_SENTENCE_END = re.compile(r"[.!?]$")

def _group_into_sentences(words):
    current_words = []
    for word in words:
        current_words.append(word)
        if _SENTENCE_END.search(word["word"]):
            sentences.append(_build_sentence(sentence_id, current_words))
            current_words = []
    if current_words:  # Remaining words without punctuation
        sentences.append(_build_sentence(sentence_id, current_words))
```

**Why word-level first?** Whisper provides word-level timestamps. Grouping into sentences preserves the exact start/end time of each sentence because we use the first word's `start` and last word's `end`.

**Edge case:** If narration has no punctuation (informal speech), everything becomes one giant sentence. The `if current_words` fallback handles this.

---

### 4.3 SDK Polymorphism (Dict vs Object Handling)

```python
def _get(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
```

**Why:** Groq SDK versions return words as either `dict` (`{"word": "hello"}`) or objects (`Word(word="hello")`). This adapter handles both transparently, making the code resilient to SDK updates.

**How to learn this:**
- Duck typing in Python
- Adapter pattern from GoF

---

## 5. Video Rendering & FFmpeg

### 5.1 Two-Pass Rendering Architecture

```
Pass 1: Render each clip individually (with speed + audio sync)
    clip_0000.mp4, clip_0001.mp4, ...

Pass 2: Concatenate all clips (stream copy, no re-encode)
    ffmpeg -f concat -safe 0 -i concat.txt -c copy output.mp4
```

**Why two passes?** Each clip needs different speed adjustment. A single-pass complex filter graph would be fragile and hard to debug. Two-pass lets each clip's encoding be independent and testable.

**Stream copy in Pass 2:** The `-c copy` flag copies encoded data without re-encoding. This is ~100× faster than re-encoding while preserving quality bit-for-bit.

---

### 5.2 Speed Adjustment via PTS Manipulation

```python
filters = [f"setpts=PTS/{effective_speed:.4f}"]  # e.g., setpts=PTS/1.5000
```

**What PTS is:** Presentation Timestamp — tells the player *when* to show each frame. Dividing by 1.5 makes each frame appear 1.5× sooner → video plays 1.5× faster.

**Speed clamping:**
```python
effective_speed = max(min_speed, min(max_speed, raw_speed))
# Ensures 0.5x ≤ speed ≤ 2.5x
```

**Why clamp?** Below 0.5× looks like a frozen slideshow. Above 2.5× is unwatchably fast. The renderer auto-freezes when speed exceeds 2.5×.

---

### 5.3 Ken Burns Effect (Subtle Zoom on Freeze Frames)

```python
total_frames = max(int(audio_duration * fps), 1)
zoom_step = (max_zoom - 1.0) / max(total_frames, 1)  # e.g., 0.03/90 = 0.000333

video_filter = (
    f"zoompan=z='min(zoom+{zoom_step:.6f},{max_zoom:.4f})'"
    f":d={total_frames}"
    f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    f":s={width}x{height}:fps={fps:.2f}"
)
```

**What it does:** Over a 3-second freeze clip at 30fps (90 frames), zooms from 1.0× to 1.03× centered on the frame. The viewer barely notices the zoom, but the screen feels "alive" instead of frozen.

**Math:** `x='iw/2-(iw/zoom/2)'` and `y='ih/2-(ih/zoom/2)'` keep the zoom centered. As `zoom` increases, the crop region shrinks equally from all sides.

**How to learn this:**
- FFmpeg docs: `zoompan` filter
- "The Ken Burns Effect" documentary technique (pan/zoom over still photos)

---

### 5.4 Silence Generation

```python
"-f", "lavfi", "-t", f"{gap_duration:.3f}",
"-i", "anullsrc=r=44100:cl=stereo"
```

**What `anullsrc` is:** A virtual audio source that generates silence. Used for gap clips where the narrator pauses — the video continues playing but there's no voiceover.

**Why not just omit audio?** The concat demuxer requires all clips to have both video and audio streams. A clip without audio would break concatenation.

---

### 5.5 Frame Padding (tpad for Slow Clips)

```python
if effective_speed < min_speed:
    pad = audio_duration - (trim_duration / effective_speed)
    filters.append(f"tpad=stop_mode=clone:stop_duration={pad:.3f}")
```

**What it does:** When the video is too short for the audio (e.g., 1s video for 5s audio), `tpad` clones the last frame to fill the remaining time. `stop_mode=clone` means "repeat the final frame" rather than showing black.

---

### 5.6 Center-Crop in Time (for Fast Clips)

```python
if effective_speed > max_speed:
    usable = audio_duration * max_speed
    trim_start = clip.source_start + (video_duration - usable) / 2
    trim_duration = usable
```

**What it does:** If 10s of video needs to play in 2s (5× speed, above 2.5× max), it takes the middle 5s of video and plays at 2.5× instead. The start and end of the segment are trimmed equally — center-cropping in the time dimension.

---

### 5.7 Output Validation via ffprobe

```python
def _validate_output(output_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-show_entries", "stream=codec_type", "-print_format", "json", str(output_path)],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    codec_types = {s["codec_type"] for s in data["streams"]}
    if "video" not in codec_types: raise RenderError("Missing video stream")
    if "audio" not in codec_types: raise RenderError("Missing audio stream")
```

**Why validate?** FFmpeg can silently produce corrupted output (missing stream, 0-second duration) if filter chains have subtle errors. Probing the output catches these before the user downloads a broken file.

---

## 6. Timeline & Sync Algorithms

### 6.1 Constrained Proportional Allocation

**The core algorithm** that distributes segment video time across multiple sentences:

```python
# Each sentence gets video proportional to its audio duration
for _, aud_dur in sent_list:
    proportion = aud_dur / total_audio
    slice_dur = seg_dur * proportion
    if use_minimums:
        slice_dur = max(slice_dur, min(_MIN_CLIP_VIDEO_DURATION, seg_dur))
    allocations.append(slice_dur)

# Normalize if allocations overflow
alloc_total = sum(allocations)
if alloc_total > seg_dur * 1.001:
    scale = seg_dur / alloc_total
    allocations = [a * scale for a in allocations]
```

**What this solves:** When 5 sentences all map to the same 10s segment, each gets 2s of video. The video advances sequentially (0-2s, 2-4s, 4-6s...) rather than replaying from 0 for each sentence.

**Overflow guard:** If minimum enforcement (`_MIN_CLIP_VIDEO_DURATION=1.0s`) causes allocations to exceed the segment, the code scales everything down proportionally, maintaining relative sizes.

---

### 6.2 Audio > Video Guard

```python
min_total = sum(
    max(seg_dur * (dur / total_audio), min(_MIN_CLIP_VIDEO_DURATION, seg_dur))
    for _, dur in sent_list
)
use_minimums = min_total <= seg_dur * 1.05  # Within 5% tolerance
```

**Problem:** 135s audio over 40s video. With 1s minimums, 20 sentences need 20s minimum. But proportional allocation only gives each ~2s. With minimums enforced, the cursor overflows — the video "restarts" mid-segment.

**Solution:** Detect overflow *before* allocating. If minimums would overflow, disable them entirely. The video plays at a consistent ~0.3× speed rather than freezing randomly.

---

### 6.3 Auto-Freeze Guard

```python
force_freeze = vid_start >= vid_end and not mapping.freeze
if force_freeze:
    logger.debug("Auto-freezing sentence %d (video exhausted at %.3fs)", ...)
```

**Why:** If a segment's video is fully consumed by earlier sentences, later sentences get zero-duration clips (start == end). Rather than crash FFmpeg, the clip auto-converts to freeze — holding the last visible frame while the voiceover continues.

---

## 7. Backend & API Design

### 7.1 FastAPI REST API with Versioned Routes

```
app/api/v1/
├── upload.py   → POST /api/v1/upload
├── process.py  → POST /api/v1/process/{job_id}
├── status.py   → GET  /api/v1/status/{job_id}
├── download.py → GET  /api/v1/download/{job_id}
├── jobs.py     → GET  /api/v1/jobs
├── preview.py  → GET  /api/v1/preview/{job_id}
└── events.py   → GET  /api/v1/events/{job_id} (SSE)
```

**API versioning via URL prefix** (`/api/v1/`): Allows breaking changes in v2 without affecting existing clients.

### 7.2 Typer CLI with Rich Console

```python
app = typer.Typer(help="Showtime — turn rough recordings into polished demos.")

@app.command()
def process(
    video: Path = typer.Argument(...),
    audio: Path = typer.Argument(...),
    output: Path = typer.Option("output.mp4", "--output", "-o"),
):
    console.print(f"[bold cyan][{progress:3d}%][/] {step}...")
```

**Why Typer + Rich?** Typer auto-generates `--help` from argument annotations. Rich provides colored, formatted terminal output with zero boilerplate.

---

## 8. Testing Strategies

### 8.1 Synthetic Test Fixtures (Programmatic Video/Audio Generation)

```python
@pytest.fixture(scope="session")
def sample_video(work_dir):
    """Generate a 3-second synthetic video with 3 color scenes."""
    colors = [(0, 0, 200), (0, 200, 0), (200, 0, 0)]  # BGR: red, green, blue
    for color in colors:
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        for _ in range(fps):
            writer.write(frame)

@pytest.fixture(scope="session")
def sample_audio(work_dir):
    """Generate a 3-second sine-wave WAV file."""
    samples = b"".join(
        struct.pack("<h", int(32767 * np.sin(2 * np.pi * freq * i / sample_rate)))
        for i in range(n_samples)
    )
```

**Why synthetic?** Real video/audio files would bloat the repo and be non-deterministic. Synthetic fixtures are tiny, deterministic, and test exactly the properties needed (3 color changes → 3 scenes).

**`scope="session"`:** Generated once per test session, not per test. Avoids re-creating videos for every test function.

### 8.2 Mock-Based Subprocess Testing

```python
def _mock_subprocess(mocker):
    mock_run = mocker.patch("app.services.renderer.subprocess.run")
    def _side_effect(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        if cmd[0] == "ffprobe":
            result.stdout = json.dumps({"streams": [...]})
        return result
    mock_run.side_effect = _side_effect
```

**Why mock FFmpeg?** FFmpeg tests would be slow (encoding video) and require FFmpeg installed. Mocking verifies the *correct commands* are built without actually running them.

### 8.3 Integration vs Unit Test Separation

```python
@pytest.mark.integration  # Only runs with: pytest -m integration
class TestRenderIntegration:
    def test_renders_valid_mp4(self, sample_video, sample_audio, tmp_path): ...

class TestBuildFfmpegClipArgs:  # Runs always — no external deps
    def test_has_two_inputs(self): ...
```

---

## 9. Data Modeling & Validation

### 9.1 Pydantic Models with Field Constraints

```python
class MappingEntry(BaseModel):
    speed_factor: float = Field(default=1.0, ge=0.25, le=4.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
```

**`ge`/`le` constraints:** Pydantic rejects invalid values at construction time. If an LLM returns `speed_factor: -1.0`, Pydantic raises `ValidationError` immediately — no silent corruption.

### 9.2 Computed Properties via @property

```python
class TimelineClip(BaseModel):
    audio_start: float
    audio_end: float

    @property
    def rendered_duration(self) -> float:
        return self.audio_end - self.audio_start
```

**Why not a stored field?** If `audio_start` or `audio_end` changed, a stored `duration` field would be stale. The property recalculates every access — always correct.

### 9.3 pydantic-settings for Environment Variables

```python
class Settings(BaseSettings):
    groq_api_key: str = ""
    scene_threshold: float = 30.0
    model_config = {"env_prefix": "SHOWTIME_", "env_file": ".env"}
```

**How it works:** `SHOWTIME_SCENE_THRESHOLD=25.0` in the environment automatically overrides the default `30.0`. No manual `os.getenv()` parsing.

---

## 10. Configuration & DevOps

### 10.1 UV Package Manager

```bash
uv pip install -e ".[dev]"    # Editable install with dev deps
uv run pytest                  # Run in the project's venv
uv run showtime video.mp4 audio.mp3
```

**Why UV over pip?** UV is 10-100× faster for dependency resolution and installation. Written in Rust.

### 10.2 Temporary Directory Pattern

```python
with tempfile.TemporaryDirectory(prefix="showtime_render_") as tmp_dir:
    work_dir = Path(tmp_dir)
    # ... render clips into work_dir ...
# Automatically cleaned up when context manager exits
```

**Why:** Intermediate clip files can be hundreds of MB. Auto-cleanup prevents disk bloat even if the process crashes.

---

## 11. Non-Obvious Tricks & Optimizations

### 11.1 The 5% Tolerance Trick
```python
use_minimums = min_total <= seg_dur * 1.05
```
Allows minimums to slightly overshoot (by 5%) — the normalization step fixes it. This prevents unnecessary fallback to pure-proportional mode for near-boundary cases.

### 11.2 OCR Deduplication + Garbage Filtering
```python
alnum_count = sum(1 for c in line if c.isalnum() or c.isspace())
if alnum_count / len(line) < 0.5: continue  # Skip "|---|---|" table borders
if len(line) < 3: continue                   # Skip "×" close buttons
if line.lower() in seen: continue             # Skip duplicate menu items
```

### 11.3 Evenly-Spaced Image Selection
```python
step = len(segments) / _MAX_VISION_IMAGES
image_indices = {int(i * step) for i in range(_MAX_VISION_IMAGES)}
```
When 12 segments exist but only 5 images allowed, picks indices [0, 2, 4, 7, 9] — evenly distributed coverage, not just the first 5.

### 11.4 Backward-Compatibility Aliases
```python
def _fallback_time_mapping(segments, sentences):
    """Alias for _scene_aware_mapping (test compatibility)."""
    return _scene_aware_mapping(segments, sentences)
```
When internal functions get renamed during refactoring, aliases prevent existing tests from breaking. Zero cost, maximum stability.

### 11.5 Robustness via `getattr` with Default
```python
tag_info = f", tag={seg.semantic_tag}" if getattr(seg, "semantic_tag", None) else ""
```
Works with both `VideoSegment` (has `semantic_tag`) and `CaptionedSegment` (might not). Avoids `AttributeError` entirely.

### 11.6 Short-Video Merge Skip
```python
total_dur = segments[-1].end - segments[0].start
if total_dur < 10.0:
    return segments  # Don't merge — every segment matters in short videos
```
A 5-second video with 3 segments: merging would destroy the structure. Only merge for longer videos where tiny segments are noise.

---

## Summary of Complexity Distribution

| Component | Lines | Key Algorithms |
|-----------|-------|---------------|
| Scene Detector | 523 | Dual-method boundary detection, 3-pass refinement, AI verification |
| AI Mapper | 608 | 3-tier strategy, self-critique, batched vision, chronological enforcement |
| Renderer | 469 | Speed clamping, Ken Burns, gap-freeze, two-pass concat |
| Timeline | 243 | Constrained proportional allocation, overflow guard, auto-freeze |
| Frame Captioner | 181 | Canny edges, contour analysis, dominant colors |
| Audio Analyzer | 195 | Dual-provider, word→sentence grouping |
| Domain Models | 128 | Pydantic constraints, computed properties |
| Pipeline | 84 | Observer pattern, orchestration |
| Config | 52 | pydantic-settings, env prefix |
| Vision Utils | 98 | Image resize, base64 batching |
| CLI | 68 | Typer + Rich |

**Total production code: ~2,649 lines across 11 modules.**
**Test code: ~165 tests across 10 test files.**
