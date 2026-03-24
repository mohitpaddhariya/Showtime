"""Status endpoint — check job progress."""

from fastapi import APIRouter, HTTPException

from app.api.jobs import job_store
from app.models.schemas import JobStatus

router = APIRouter()


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get the current status, progress, and result (if completed) of a job.

    Frontend only needs to poll this single endpoint — it returns everything
    including download_url and metadata when the job is done.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    response = JobStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        step=job.step,
        error=job.error,
        created_at=job.created_at,
        video_filename=job.video_filename,
        audio_filename=job.audio_filename,
    )

    # Include result metadata when completed
    if job.status == "completed" and job.result:
        response.download_url = f"/api/v1/download/{job_id}"
        response.preview_url = f"/api/v1/preview/{job_id}"
        response.duration = job.result.duration
        response.segments_detected = job.result.segments_detected
        response.sentences_detected = job.result.sentences_detected
        response.clips_rendered = job.result.clips_rendered

    # Preview available as soon as video is uploaded
    if job.video_path and job.video_path.exists():
        response.preview_url = f"/api/v1/preview/{job_id}"

    return response
