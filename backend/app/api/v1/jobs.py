"""Jobs endpoint — list, delete, and manage jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.jobs import job_store
from app.models.schemas import JobListResponse, JobStatus

router = APIRouter()


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all jobs, most recent first."""
    jobs = job_store.list_all()
    items = []
    for job in jobs:
        status = JobStatus(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            step=job.step,
            error=job.error,
            created_at=job.created_at,
            video_filename=job.video_filename,
            audio_filename=job.audio_filename,
        )
        if job.status == "completed" and job.result:
            status.download_url = f"/api/v1/download/{job.job_id}"
            status.duration = job.result.duration
            status.segments_detected = job.result.segments_detected
            status.sentences_detected = job.result.sentences_detected
            status.clips_rendered = job.result.clips_rendered
            status.preview_url = f"/api/v1/preview/{job.job_id}"
        items.append(status)
    return JobListResponse(jobs=items, total=len(items))


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files. Cannot delete a job that is currently processing."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status == "processing":
        raise HTTPException(409, "Cannot delete a job that is currently processing")

    job_store.delete(job_id)
    return {"message": f"Job {job_id} deleted"}
