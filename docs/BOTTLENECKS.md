# Showtime — Bottlenecks & Limits

How each pipeline step performs, what limits it, and how to fix it.

---

## Pipeline Step Performance

| Step | Tool | 1 min video | 5 min video | 10 min video | 30 min video |
|------|------|-------------|-------------|--------------|--------------|
| Scene Detection | OpenCV | ~2s | ~8s | ~15s | ~45s |
| AI Scene Verify | Groq Vision (chunks of 5) | ~1s | ~2s | ~3s | ~5s |
| Frame Captioning | Tesseract | ~1s | ~3s | ~5s | ~15s |
| Transcription (Groq) | Groq Whisper API | ~1s | ~3s | ~5s | ~10s |
| Transcription (local) | Whisper base/CPU | ~30s | ~2.5min | ~5min | ~15min |
| AI Mapping | Groq Vision (1 call) | ~1s | ~2s | ~2s | ~3s |
| Rendering | FFmpeg | ~5s | ~20s | ~40s | ~2min |
| **Total (Groq)** | | **~11s** | **~38s** | **~1.1min** | **~3.3min** |
| **Total (local)** | | **~40s** | **~3min** | **~6min** | **~18min** |

---

## Bottleneck 1: Audio Transcription (biggest)

### Problem
Local Whisper on CPU is the slowest step by far. The `base` model processes at ~0.5x realtime on Apple Silicon, meaning a 10-minute audio takes ~20 minutes. The `large` model is even slower (~0.1x realtime).

### Solutions (implemented)

**Use Groq Whisper API (default):**
```bash
SHOWTIME_WHISPER_PROVIDER=groq   # default
```
- Model: `whisper-large-v3-turbo` (best quality, still fast)
- Speed: ~10x realtime (10 min audio transcribed in ~1 second)
- Free tier: 28,800 seconds of audio/day = **480 minutes/day**
- Quality: Better than local `base` model (uses `large-v3` on Groq's hardware)

**Use local Whisper (offline):**
```bash
SHOWTIME_WHISPER_PROVIDER=local
SHOWTIME_WHISPER_MODEL=tiny    # fastest, lower quality
```

### Local Whisper model comparison

| Model | Size | Speed (CPU) | Speed (GPU) | Quality |
|-------|------|-------------|-------------|---------|
| tiny | 39M | ~10x realtime | ~32x | Low -- misses words |
| base | 74M | ~5x realtime | ~16x | OK for clear English |
| small | 244M | ~2x realtime | ~6x | Good |
| medium | 769M | ~0.5x realtime | ~2x | Great |
| large | 1.5GB | ~0.1x realtime | ~1x | Best |

Recommendation: Use `tiny` or `base` for development, Groq for production.

---

## Bottleneck 2: Scene Detection (OpenCV + AI Verify)

### Problem
Sampling every frame is expensive for long/high-fps videos. A 30 min video at 30fps = 54,000 frames. Also, single-threshold detection misses subtle transitions (scrolling, tab switches).

### Current mitigation (implemented)
- `SHOWTIME_SAMPLE_FPS=2` -- samples 2 frames/sec (15x fewer frames)
- **Auto-split**: segments > `MAX_SEGMENT_DURATION` (8s) re-scanned at lower threshold (20 vs 30)
- **Merge**: segments < `MIN_SEGMENT_DURATION` (1.5s) absorbed into neighbors (removes cursor-blink noise)
- **[v2] AI Verification**: Llama 4 Scout confirms/merges/splits segments semantically. Processes in chunks of 5 images (Groq's per-request limit). Adds semantic tags.
- Result: a 37s demo video --> 5 meaningful segments (was 3 without refinement, 21 before merge)

### Tuning
```bash
SHOWTIME_SCENE_THRESHOLD=30          # primary threshold (higher = fewer scenes)
SHOWTIME_SCENE_REFINE_THRESHOLD=20   # secondary threshold for long segments
SHOWTIME_MAX_SEGMENT_DURATION=8      # auto-split above this
SHOWTIME_MIN_SEGMENT_DURATION=1.5    # merge below this
SHOWTIME_SAMPLE_FPS=1                # faster, might miss quick transitions
SHOWTIME_AI_VERIFY_SCENES=false      # skip AI verification to save 1+ API calls
```

---

## Bottleneck 3: FFmpeg Rendering

### Problem
Two-pass rendering means each clip is encoded separately, then concatenated. Encoding is CPU-bound.

### Current settings
```bash
SHOWTIME_OUTPUT_PRESET=medium  # balance of speed vs compression
SHOWTIME_CRF=23               # quality (lower = better quality, slower)
```

### v2 rendering improvements
- **Ken Burns on freeze clips**: subtle zoom (1.0x-->1.03x) prevents dead-screen effect. Minimal CPU overhead since it's a simple zoompan filter.
- **Gap-freeze**: gap clips with insufficient video now hold a still frame instead of encoding extreme slow-motion. Faster to render (single frame loop vs. speed-adjusted video).

### Tuning for speed
```bash
SHOWTIME_OUTPUT_PRESET=ultrafast  # 5-10x faster encoding, larger files
SHOWTIME_CRF=28                  # lower quality, faster
SHOWTIME_KEN_BURNS_ON_FREEZE=false  # skip Ken Burns for faster freeze clips
```

### Tuning for quality
```bash
SHOWTIME_OUTPUT_PRESET=slow       # best compression, slowest
SHOWTIME_CRF=18                  # near-lossless
SHOWTIME_KEN_BURNS_ZOOM=1.05     # more noticeable zoom effect
```

### Future improvements
- Hardware acceleration: `-c:v h264_videotoolbox` on macOS (10x faster)
- Single-pass filter graph instead of two-pass (avoids intermediate files)
- GPU encoding with NVENC on Linux

---

## Bottleneck 4: Groq API Rate Limits

### Free tier limits

| Resource | Limit | v2 usage per video | Headroom |
|----------|-------|--------------------|----------|
| Vision model (Llama 4 Scout) | 15/min, 7,000/day | 1-2 (verify) + 1 (map) = 2-3 | Plenty |
| Text model (Llama 3.3 70B) | 30/min, 14,400/day | 0-1 (refinement only) | Plenty |
| Whisper requests | 20/min, 7,200/day | 1 per video | Plenty |
| Whisper audio | 28,800 sec/day | = 480 min/day of audio | Plenty |

### v2 Groq call budget (per video)

| Step | Calls | Model |
|------|-------|-------|
| Whisper transcription | 1 | whisper-large-v3-turbo |
| AI scene verification | ceil(segments / 5) | Llama 4 Scout (vision) |
| AI mapping | 1 | Llama 4 Scout (vision) |
| Refinement (optional) | 0-1 | Llama 3.3 70B (text) |
| **Total** | **3-5** | |

**Note:** AI scene verification processes in chunks of 5 images (Groq's limit). A video with 12 segments needs 3 verification calls. Set `SHOWTIME_AI_VERIFY_SCENES=false` to skip.

### Image limit per request

Groq Llama 4 Scout allows **max 5 images per API request**. The pipeline handles this:
- **Scene verification**: chunks segments into batches of 5
- **AI mapping**: picks 5 evenly-spaced keyframes as images, includes the rest as text-only descriptions

### Max video length on free tier

The Groq Whisper API has a **25 MB file size limit** per request. At typical audio bitrates:

| Audio format | Bitrate | 25 MB = |
|-------------|---------|---------|
| WAV 16kHz mono | 256 kbps | ~13 min |
| MP3 128kbps | 128 kbps | ~26 min |
| MP3 64kbps | 64 kbps | ~52 min |

**For longer audio:** Convert to MP3 first to get more minutes per request, or split into chunks.

### LLM context for long videos

| Video length | Segments | Sentences | LLM tokens | Fits in 128K context? |
|-------------|----------|-----------|------------|----------------------|
| 5 min | 10-15 | 15-20 | ~2,400 | Yes |
| 15 min | 25-40 | 40-60 | ~5,500 | Yes |
| 30 min | 50-80 | 80-120 | ~11,000 | Yes |
| 60 min | 100+ | 150+ | ~20,000 | Yes |

LLM context is not a limit even for 1hr+ videos. However, videos with >25 segments will need multiple vision verification calls (5 images each).

---

## Bottleneck 5: Memory

### Problem
OpenCV loads frames into RAM. Very long or high-resolution videos can consume significant memory.

### Current mitigation
- Frame sampling (`SHOWTIME_SAMPLE_FPS=2`) reduces frames loaded
- Frames are converted to grayscale for comparison (1/3 memory vs color)
- Keyframes are saved to disk, not held in memory
- [v2] Keyframes resized to 1024px max before base64 encoding (reduces API payload)

### Rough memory usage

| Video | Resolution | Memory |
|-------|-----------|--------|
| 5 min | 1080p | ~200 MB |
| 15 min | 1080p | ~500 MB |
| 30 min | 1080p | ~1 GB |
| 30 min | 4K | ~4 GB |

### Future improvements
- Stream frames instead of collecting all in a list
- Downsample to 480p for scene detection (full res not needed for pixel diffs)

---

## Bottleneck 6: Tesseract OCR

### Problem
OCR is run on each keyframe. For videos with many scenes, this adds up (~0.5s per frame).

### Current behavior
- Only one keyframe per scene is processed (not every frame)
- Simple screen recordings typically have 5-30 scenes
- [v2] OCR descriptions are supplementary -- the vision mapper primarily uses actual keyframe images, so OCR quality is less critical

### Tuning
For faster OCR with slightly lower accuracy:
```python
pytesseract.image_to_string(image, config='--oem 3 --psm 6')
```

### Future improvements
- Replace Tesseract with a vision LLM (e.g., send keyframes to Groq's multimodal model)
- Parallel OCR processing with ThreadPoolExecutor

---

## Quick Reference: Environment Variables for Tuning

```bash
# ── Speed optimizations ──
SHOWTIME_WHISPER_PROVIDER=groq         # 10-50x faster than local
SHOWTIME_SAMPLE_FPS=1                  # fewer frames to process
SHOWTIME_OUTPUT_PRESET=ultrafast       # fast encoding
SHOWTIME_CRF=28                        # lower quality, faster
SHOWTIME_AI_VERIFY_SCENES=false        # skip AI scene verification
SHOWTIME_KEN_BURNS_ON_FREEZE=false     # skip Ken Burns effect

# ── Quality optimizations ──
SHOWTIME_WHISPER_PROVIDER=groq         # uses large-v3 model
SHOWTIME_GROQ_WHISPER_MODEL=whisper-large-v3
SHOWTIME_SAMPLE_FPS=4                  # catch more scene changes
SHOWTIME_OUTPUT_PRESET=slow            # best compression
SHOWTIME_CRF=18                        # near-lossless
SHOWTIME_KEN_BURNS_ZOOM=1.05           # more visible Ken Burns
SHOWTIME_PACING_THRESHOLD=9.0          # trigger refinement more often
```

---

## Bottleneck 7: File Format Compatibility

### Supported formats

| Type | Upload API | Groq Whisper | Local Whisper | FFmpeg Renderer |
|------|-----------|-------------|--------------|-----------------|
| **Video** | `.mp4 .mov .avi .mkv .webm .flv .wmv` | N/A | N/A | Any FFmpeg format |
| **Audio** | `.mp3 .wav .m4a .aac .ogg .flac .webm .wma .mpga` | `mp3 mp4 mpeg mpga m4a wav webm` | Any FFmpeg format | Any FFmpeg format |

### Groq Whisper 25 MB file limit

| Format | Bitrate | Max duration in 25 MB |
|--------|---------|----------------------|
| WAV 16kHz mono | 256 kbps | ~13 min |
| MP3 128 kbps | 128 kbps | ~26 min |
| MP3 64 kbps | 64 kbps | ~52 min |
| FLAC (compressed) | ~500 kbps | ~7 min |
| AAC 96 kbps | 96 kbps | ~35 min |

**Tip:** For long audio, convert to compressed MP3 first:
```bash
ffmpeg -i long_audio.wav -b:a 64k long_audio.mp3
```

---

## Summary: Recommended Setup

| Use case | Whisper | LLM | Preset | AI Verify | Processing time (10 min video) |
|----------|---------|-----|--------|-----------|-------------------------------|
| **Fast dev** | Groq | Groq | ultrafast | off | ~30s |
| **Production** | Groq | Groq | medium | on | ~1 min |
| **Offline** | local (base) | Ollama | medium | off | ~6 min |
| **Best quality** | Groq (large-v3) | Groq (70b) | slow | on | ~2 min |
