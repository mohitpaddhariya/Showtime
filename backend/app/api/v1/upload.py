"""Upload endpoint — accepts video + audio files."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from tempfile import mkdtemp

from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from app.api.jobs import job_store
from app.core.exceptions import ShowtimeError
from app.models.schemas import UploadResponse, PipelineInput, JobStatus
from app.services.pipeline import run_pipeline

router = APIRouter()

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

SUPPORTED_VIDEO = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv")
SUPPORTED_AUDIO = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".webm", ".wma", ".mpga")


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    video: UploadFile = File(..., description="Screen recording"),
    audio: UploadFile = File(..., description="Voiceover audio"),
):
    """Upload video + audio files. Returns job_id for use with /process or /upload-and-process."""
    video_path, audio_path, work_dir = await _save_uploads(video, audio)
    output_path = work_dir / "output.mp4"
    job = job_store.create(video_path, audio_path, output_path, work_dir)
    job_store.update(job.job_id, video_filename=video.filename, audio_filename=audio.filename)

    return UploadResponse(
        job_id=job.job_id,
        message="Files uploaded. POST /api/v1/process/{job_id} to start, or use /api/v1/upload-and-process for one step.",
    )


@router.post("/upload-and-process", response_model=JobStatus)
async def upload_and_process(
    video: UploadFile = File(..., description="Screen recording"),
    audio: UploadFile = File(..., description="Voiceover audio"),
):
    """Upload + immediately start processing in one call. Returns job status to poll."""
    video_path, audio_path, work_dir = await _save_uploads(video, audio)
    output_path = work_dir / "output.mp4"
    job = job_store.create(video_path, audio_path, output_path, work_dir)
    job_store.update(job.job_id, video_filename=video.filename, audio_filename=audio.filename)

    # Start processing immediately
    job_store.update(job.job_id, status="processing", progress=0, step="Starting")

    def _run():
        try:
            pipeline_input = PipelineInput(
                video_path=job.video_path,
                audio_path=job.audio_path,
                output_path=job.output_path,
                work_dir=job.work_dir,
            )

            def _on_progress(step: str, progress: int):
                job_store.update(job.job_id, step=step, progress=progress)

            result = run_pipeline(pipeline_input, on_progress=_on_progress)
            job_store.update(job.job_id, status="completed", progress=100, result=result)
        except ShowtimeError as e:
            job_store.update(job.job_id, status="failed", error=str(e))
        except Exception as e:
            job_store.update(job.job_id, status="failed", error=f"Unexpected error: {e}")

    threading.Thread(target=_run, daemon=True).start()

    return JobStatus(
        job_id=job.job_id,
        status="processing",
        progress=0,
        step="Starting",
    )


async def _save_uploads(
    video: UploadFile,
    audio: UploadFile,
) -> tuple[Path, Path, Path]:
    """Validate and save uploaded files. Returns (video_path, audio_path, work_dir)."""
    video_ext = Path(video.filename or "").suffix.lower()
    audio_ext = Path(audio.filename or "").suffix.lower()

    if video_ext not in SUPPORTED_VIDEO:
        raise HTTPException(400, f"Unsupported video format: {video_ext}. Supported: {', '.join(SUPPORTED_VIDEO)}")
    if audio_ext not in SUPPORTED_AUDIO:
        raise HTTPException(400, f"Unsupported audio format: {audio_ext}. Supported: {', '.join(SUPPORTED_AUDIO)}")

    # Check file sizes
    video_size = video.size or 0
    audio_size = audio.size or 0
    if video_size > MAX_FILE_SIZE:
        raise HTTPException(413, f"Video too large ({video_size // 1024 // 1024}MB). Max: {MAX_FILE_SIZE // 1024 // 1024}MB")
    if audio_size > MAX_FILE_SIZE:
        raise HTTPException(413, f"Audio too large ({audio_size // 1024 // 1024}MB). Max: {MAX_FILE_SIZE // 1024 // 1024}MB")

    work_dir = Path(mkdtemp(prefix="showtime_"))
    job_dir = work_dir / "input"
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / f"video{video_ext}"
    audio_path = job_dir / f"audio{audio_ext}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    return video_path, audio_path, work_dir
