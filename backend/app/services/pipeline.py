"""Pipeline orchestration — runs all steps in a clean, linear sequence.

Groq API call budget for a typical 5-minute video:
  1. Whisper transcription (audio_analyzer)   — 1 Groq call
  2. AI scene verification (scene_detector)   — 1 Groq vision call (optional)
  3. AI mapping (ai_mapper)                   — 1 Groq vision call
  4. Refinement (ai_mapper, if needed)        — 0-1 Groq text call
  ─────────────────────────────────────────────────────
  Total: 3-4 Groq calls (down from 5+ in v1)

All other steps (OpenCV scene detection, Tesseract OCR, FFmpeg rendering)
are local and use zero API calls.
"""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.models.domain import CaptionedSegment, MappingEntry, VoiceoverSentence
from app.models.schemas import PipelineInput, PipelineResult
from app.services.ai_mapper import map_sentences_to_segments
from app.services.audio_analyzer import transcribe_audio
from app.services.frame_captioner import caption_segments
from app.services.renderer import render
from app.services.scene_detector import detect_scenes
from app.services.timeline import assemble_timeline

logger = logging.getLogger(__name__)


def run_pipeline(
    pipeline_input: PipelineInput,
    settings: Settings | None = None,
    on_progress: callable | None = None,
) -> PipelineResult:
    """Execute the full Showtime pipeline.

    Steps:
      1. Scene detection (OpenCV + optional AI verification)  [10%]
      2. Frame captioning (Tesseract OCR, local)              [25%]
      3. Audio transcription (Groq Whisper)                   [40%]
      4. AI mapping (Llama 4 Scout vision + optional refine)  [60%]
      5. Timeline assembly (local computation)                [75%]
      6. Video rendering (FFmpeg, local)                      [85%]

    Args:
        pipeline_input: Validated paths for video, audio, output, work dir.
        settings: Optional settings override.
        on_progress: Optional callback(step: str, progress: int) for UI updates.

    Returns:
        PipelineResult with stats about the rendered video.
    """
    if settings is None:
        settings = Settings()

    def _notify(step: str, progress: int) -> None:
        if on_progress:
            on_progress(step, progress)
        logger.info("[%d%%] %s", progress, step)

    # ── Step 1: Scene detection ──────────────────────────────────
    # OpenCV pixel-diff analysis + optional Llama 4 Scout verification.
    # Groq calls: 0 (AI verify off) or 1 (AI verify on).
    _notify("Detecting scenes", 10)
    all_segments = detect_scenes(
        pipeline_input.video_path, pipeline_input.work_dir, settings,
    )

    # Filter out idle/dead segments — removes boring screen-recording gaps
    segments = [s for s in all_segments if not s.is_idle]
    idle_count = len(all_segments) - len(segments)
    logger.info(
        "Scene detection: %d segments (%d idle removed), tags: %s",
        len(segments), idle_count,
        [s.semantic_tag for s in segments if s.semantic_tag],
    )

    # ── Step 2: Frame captioning ─────────────────────────────────
    # Tesseract OCR + OpenCV structural analysis. Zero API calls.
    # Provides text descriptions as fallback context for the AI mapper
    # (vision mapper primarily uses actual images, not descriptions).
    _notify("Captioning keyframes", 25)
    captioned = caption_segments(segments)

    # ── Step 3: Audio transcription ──────────────────────────────
    # Groq Whisper: 1 API call. Returns word-level timestamps.
    _notify("Transcribing voiceover", 40)
    sentences = transcribe_audio(pipeline_input.audio_path, settings)
    logger.info("Transcription: %d sentences, %.1fs total audio",
                len(sentences), sentences[-1].end if sentences else 0)

    # ── Step 4: AI mapping ───────────────────────────────────────
    # Llama 4 Scout vision: 1 API call (all keyframes batched).
    # Optional text refinement: 0-1 API call if pacing_score < threshold.
    _notify("Mapping voiceover to segments", 60)
    mappings = map_sentences_to_segments(captioned, sentences, settings)
    freeze_count = sum(1 for m in mappings if m.freeze)
    logger.info("Mapping: %d entries, %d freezes", len(mappings), freeze_count)

    # ── Step 5: Timeline assembly ────────────────────────────────
    # Pure local computation. Zero API calls.
    _notify("Assembling timeline", 75)
    timeline = assemble_timeline(
        mappings, captioned, sentences,
        pipeline_input.video_path, pipeline_input.audio_path,
    )
    gap_count = sum(1 for c in timeline.clips if c.is_gap)
    logger.info("Timeline: %.1fs, %d clips (%d content, %d gaps)",
                timeline.total_duration, len(timeline.clips),
                len(timeline.clips) - gap_count, gap_count)

    # ── Step 6: Rendering ────────────────────────────────────────
    # FFmpeg local rendering. Zero API calls.
    _notify("Rendering final video", 85)
    render(timeline, pipeline_input.output_path, settings)

    _notify("Complete", 100)

    return PipelineResult(
        output_path=pipeline_input.output_path,
        duration=timeline.total_duration,
        segments_detected=len(segments),
        sentences_detected=len(sentences),
        clips_rendered=len(timeline.clips),
    )
