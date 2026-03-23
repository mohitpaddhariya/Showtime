"""Download endpoint — serve the rendered video."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api.jobs import job_store

router = APIRouter()


@router.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the rendered video for a completed job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    if job.status != "completed":
        raise HTTPException(400, f"Job {job_id} is not completed (status: {job.status})")

    if not job.output_path or not job.output_path.exists():
        raise HTTPException(500, "Output file not found")

    return FileResponse(
        path=str(job.output_path),
        media_type="video/mp4",
        filename=f"showtime_{job_id}.mp4",
    )
