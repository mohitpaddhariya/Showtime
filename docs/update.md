# Showtime v2 Pipeline Update

## Summary

Major refactor of the core pipeline for better accuracy, fewer API calls, and smoother video output. All 165 tests pass with zero breaking changes.

---

## What Changed

### 1. Scene Detector -- AI Verification Pass (NEW)

**Before:** OpenCV pixel-diff only. Could split a slowly-scrolling page into multiple segments.

**After:** After OpenCV detection, ALL keyframes are batched into ONE Llama 4 Scout vision call. The model confirms/merges/splits segments based on actual visual content and assigns semantic tags (e.g., "landing_page", "code_editor").

- **Files:** `app/services/scene_detector.py`, `app/services/vision_utils.py` (new)
- **Config:** `SHOWTIME_AI_VERIFY_SCENES=true` (set `false` to save 1 API call)
- **Groq calls:** +1 vision call (optional)

### 2. AI Mapper -- Single Vision Call with Self-Critique (REWRITE)

**Before:** 3-tier fallback (vision -> text -> chronological) + 2-pass refinement loop. Up to 5 Groq calls.

**After:** ONE comprehensive vision call with ALL keyframes + full transcript batched. Built-in self-critique (model rates its own pacing 0-10). Only triggers 1 refinement call if `pacing_score < 8.5`.

- **File:** `app/services/ai_mapper.py`
- **Config:** `SHOWTIME_MAX_FREEZE_COUNT=3`, `SHOWTIME_PACING_THRESHOLD=8.5`
- **Groq calls:** 1-2 (down from up to 5)
- **New fields on MappingEntry:** `confidence` (0-1), `reasoning` (string)

### 3. Timeline -- Improved Allocation (IMPROVED)

**Before:** Proportional splitting with minimum enforcement that could silently disable.

**After:** Same proportional algorithm but with normalization guard -- when minimums would overflow a segment, allocations are scaled down proportionally instead of disabling minimums entirely.

- **File:** `app/services/timeline.py`
- **Breaking changes:** None

### 4. Renderer -- Ken Burns + Gap-Freeze Fix (IMPROVED)

**Before:** Freeze clips showed a static looped frame (dead screen). Gap clips played 0.1s of video at extreme slow-mo (0.1x speed = laggy/glitchy).

**After:**
- **Ken Burns on freeze clips:** Subtle slow zoom (1.0x -> 1.03x) centered on frame. Keeps viewers engaged during static moments.
- **Gap-freeze:** When gap clips have too little video for smooth playback (speed < 0.5x), holds a clean still frame + silence instead of laggy slow-motion.

- **File:** `app/services/renderer.py`
- **Config:** `SHOWTIME_KEN_BURNS_ON_FREEZE=true`, `SHOWTIME_KEN_BURNS_ZOOM=1.03`

### 5. Pipeline -- Cleaner Orchestration (IMPROVED)

**Before:** Basic step logging with progress percentage.

**After:** Rich logging with segment counts, semantic tags, freeze counts, gap counts. Clear Groq call budget tracking in comments.

- **File:** `app/services/pipeline.py`
- **Total Groq calls per run:** 3-4 (Whisper + AI Verify + AI Map + optional Refine)

### 6. Domain Model -- New Optional Fields (BACKWARD COMPATIBLE)

- `VideoSegment.semantic_tag: str | None = None` -- AI-generated content label
- `MappingEntry.confidence: float = 1.0` -- mapping confidence score
- `MappingEntry.reasoning: str = ""` -- mapping explanation

### 7. New Helper Module

- `app/services/vision_utils.py` -- shared keyframe resizing (1024px max), base64 encoding, multimodal content block batching. Used by both scene_detector and ai_mapper.

---

## New Config Keys

| Key | Default | Purpose |
|-----|---------|---------|
| `SHOWTIME_AI_VERIFY_SCENES` | `true` | AI scene verification (1 extra Groq call) |
| `SHOWTIME_MAX_FREEZE_COUNT` | `3` | Max freeze frames per video |
| `SHOWTIME_PACING_THRESHOLD` | `8.5` | Self-critique score below which triggers refinement |
| `SHOWTIME_KEN_BURNS_ON_FREEZE` | `true` | Subtle zoom on freeze clips |
| `SHOWTIME_KEN_BURNS_ZOOM` | `1.03` | Max zoom factor (3% = very subtle) |

All have defaults. No existing config keys changed.

---

## Groq API Call Budget (per video)

| Step | Calls (v1) | Calls (v2) |
|------|-----------|-----------|
| Whisper transcription | 1 | 1 |
| AI scene verification | 0 | 0-1 |
| AI mapping | 1-3 | 1 |
| Refinement | 0-2 | 0-1 |
| **Total** | **2-6** | **3-4** |

---

## Bug Fixes

- **Gap clip laggy playback:** Gap clips with only 0.1s of video were being played at 0.1-0.26x speed, creating near-frozen slow-motion during every speech pause. Now detected and rendered as clean still frame + silence.

---

## Files Changed

| File | Action |
|------|--------|
| `app/models/domain.py` | Added optional fields (semantic_tag, confidence, reasoning) |
| `app/core/config.py` | Added 5 new settings with defaults |
| `app/services/vision_utils.py` | **NEW** -- shared vision utilities |
| `app/services/scene_detector.py` | Added AI verification pass |
| `app/services/ai_mapper.py` | Rewritten -- single vision call architecture |
| `app/services/timeline.py` | Improved allocation normalization |
| `app/services/renderer.py` | Ken Burns + gap-freeze fix |
| `app/services/pipeline.py` | Cleaner orchestration + rich logging |
| `.env` / `.env.example` | Added new config keys |
| `docs/AI_CONTEXT.md` | Updated to reflect v2 architecture |
| `pipeline.md` | Updated all 6 steps |
| `scripts/generate_diagrams.py` | Updated diagrams for v2 |

---

## Migration Guide

**Zero code changes needed.** All public APIs, domain models, and schemas are backward compatible. New `.env` keys are optional with sensible defaults. All 165 tests pass.
