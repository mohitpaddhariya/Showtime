"""Domain models for the Showtime pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


# ── Scene Detector Output ─────────────────────────────────


class VideoSegment(BaseModel):
    """A contiguous segment of the source screen recording."""

    segment_id: int
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    is_idle: bool = Field(default=False, description="True if negligible visual change")
    keyframe_path: Path | None = Field(
        default=None, description="Path to the extracted keyframe PNG"
    )
    semantic_tag: str | None = Field(
        default=None,
        description="AI-generated semantic label (e.g. 'landing_page', 'settings_modal')",
    )

    @property
    def duration(self) -> float:
        return self.end - self.start


# ── Frame Captioner Output ────────────────────────────────


class CaptionedSegment(BaseModel):
    """A VideoSegment enriched with a textual description from OCR."""

    segment_id: int
    start: float
    end: float
    is_idle: bool = False
    keyframe_path: Path | None = None
    description: str = Field(default="", description="OCR / visual description of the keyframe")

    @property
    def duration(self) -> float:
        return self.end - self.start


# ── Audio Analyzer Output ─────────────────────────────────


class VoiceoverSentence(BaseModel):
    """A single sentence from the voiceover transcription."""

    sentence_id: int
    text: str
    start: float = Field(description="Start time in seconds within the audio")
    end: float = Field(description="End time in seconds within the audio")

    @property
    def duration(self) -> float:
        return self.end - self.start


# ── AI Mapper Output ──────────────────────────────────────


class MappingEntry(BaseModel):
    """Maps one voiceover sentence to one screen segment."""

    sentence_id: int
    segment_id: int
    speed_factor: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback speed multiplier for the video segment",
    )
    freeze: bool = Field(
        default=False,
        description="If true, hold the keyframe still while audio plays (for reading/emphasis)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="AI confidence in this mapping (0.0-1.0)",
    )
    reasoning: str = Field(
        default="",
        description="AI reasoning for this mapping decision",
    )


# ── Timeline Assembly Output ──────────────────────────────


class TimelineClip(BaseModel):
    """A single clip in the final edit decision list."""

    order: int
    source_start: float = Field(description="Start in the original video (seconds)")
    source_end: float = Field(description="End in the original video (seconds)")
    speed_factor: float = Field(default=1.0)
    audio_start: float = Field(description="Start in the voiceover audio (seconds)")
    audio_end: float = Field(description="End in the voiceover audio (seconds)")
    is_gap: bool = Field(default=False, description="True if this is a silent pause between sentences")
    freeze: bool = Field(default=False, description="True if video should hold still while audio plays")

    @property
    def rendered_duration(self) -> float:
        """How long this clip will be in the final video."""
        return self.audio_end - self.audio_start


class Timeline(BaseModel):
    """The complete edit decision list for the final render."""

    clips: list[TimelineClip]
    source_video: Path
    source_audio: Path

    @property
    def total_duration(self) -> float:
        return sum(c.rendered_duration for c in self.clips)
