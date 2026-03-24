"""Preview endpoint — serve a thumbnail from the video."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api.jobs import job_store

router = APIRouter()


@router.get("/preview/{job_id}")
async def get_preview(job_id: str):
    """Get a thumbnail preview image for a job's video.

    Extracts a frame at 1 second (or first frame) from the source video.
    Cached on disk after first generation.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    if not job.video_path or not job.video_path.exists():
        raise HTTPException(400, "Source video not found")

    # Check for cached thumbnail
    thumb_path = job.work_dir / "preview.jpg" if job.work_dir else None
    if thumb_path and thumb_path.exists():
        return FileResponse(str(thumb_path), media_type="image/jpeg")

    if not thumb_path:
        raise HTTPException(500, "No work directory for job")

    # Generate thumbnail with ffmpeg
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(job.video_path),
                "-ss", "1",
                "-vframes", "1",
                "-vf", "scale=640:-1",
                "-q:v", "3",
                str(thumb_path),
            ],
            capture_output=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        raise HTTPException(500, "Failed to generate preview thumbnail")

    if not thumb_path.exists():
        raise HTTPException(500, "Failed to generate preview thumbnail")

    return FileResponse(str(thumb_path), media_type="image/jpeg")
