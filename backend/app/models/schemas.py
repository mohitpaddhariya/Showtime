"""API request/response schemas and pipeline I/O models."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


# ── Pipeline I/O ──────────────────────────────────────────


class PipelineInput(BaseModel):
    """Top-level input to the full pipeline."""

    video_path: Path
    audio_path: Path
    output_path: Path
    work_dir: Path = Field(description="Temporary working directory for intermediate files")


class PipelineResult(BaseModel):
    """Top-level output from the full pipeline."""

    output_path: Path
    duration: float
    segments_detected: int
    sentences_detected: int
    clips_rendered: int


# ── API Schemas ───────────────────────────────────────────


class UploadResponse(BaseModel):
    job_id: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str = Field(description="pending | processing | completed | failed")
    progress: int = Field(default=0, ge=0, le=100)
    step: str | None = Field(default=None, description="Current pipeline step")
    error: str | None = None
    # Populated when status=completed
    download_url: str | None = None
    duration: float | None = None
    segments_detected: int | None = None
    sentences_detected: int | None = None
    clips_rendered: int | None = None
    # Metadata
    created_at: float | None = None
    video_filename: str | None = None
    audio_filename: str | None = None
    preview_url: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobStatus]
    total: int


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
