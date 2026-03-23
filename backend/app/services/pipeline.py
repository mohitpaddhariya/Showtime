"""Pipeline orchestration — runs all 6 steps in sequence."""

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

    Args:
        pipeline_input: Validated paths for video, audio, output, and work directory.
        settings: Optional settings override.
        on_progress: Optional callback(step: str, progress: int) for status updates.

    Returns:
        PipelineResult with stats about the rendered video.
    """
    if settings is None:
        settings = Settings()

    def _notify(step: str, progress: int) -> None:
        if on_progress:
            on_progress(step, progress)

    # Step 1: Scene detection
    _notify("Detecting scenes", 10)
    all_segments = detect_scenes(pipeline_input.video_path, pipeline_input.work_dir, settings)
    # Filter out idle/dead segments — the pitch promises "cuts out boring parts"
    segments = [s for s in all_segments if not s.is_idle]
    idle_count = len(all_segments) - len(segments)
    logger.info("Detected %d scene(s), removed %d idle segment(s)", len(all_segments), idle_count)

    # Step 2: Frame captioning
    _notify("Captioning keyframes", 25)
    captioned = caption_segments(segments)

    # Step 3: Audio analysis
    _notify("Transcribing voiceover", 40)
    sentences = transcribe_audio(pipeline_input.audio_path, settings)
    logger.info("Found %d sentence(s)", len(sentences))

    # Step 4: AI mapping
    _notify("Mapping voiceover to segments", 60)
    mappings = map_sentences_to_segments(captioned, sentences, settings)

    # Step 5: Timeline assembly
    _notify("Assembling timeline", 75)
    timeline = assemble_timeline(
        mappings, captioned, sentences,
        pipeline_input.video_path, pipeline_input.audio_path,
    )
    logger.info("Timeline: %.1fs across %d clip(s)", timeline.total_duration, len(timeline.clips))

    # Step 6: Rendering
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
