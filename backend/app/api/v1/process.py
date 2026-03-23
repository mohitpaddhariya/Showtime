"""Process endpoint — triggers the pipeline for a job."""

from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException

from app.api.jobs import job_store
from app.core.exceptions import ShowtimeError
from app.models.schemas import PipelineInput, JobStatus
from app.services.pipeline import run_pipeline

router = APIRouter()


@router.post("/process/{job_id}", response_model=JobStatus)
async def process_job(job_id: str):
    """Start processing a previously uploaded job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    if job.status == "processing":
        raise HTTPException(409, f"Job {job_id} is already processing")

    if job.status == "completed":
        raise HTTPException(409, f"Job {job_id} is already completed")

    # Run pipeline in background thread
    job_store.update(job_id, status="processing", progress=0, step="Starting")

    def _run():
        try:
            pipeline_input = PipelineInput(
                video_path=job.video_path,
                audio_path=job.audio_path,
                output_path=job.output_path,
                work_dir=job.work_dir,
            )

            def _on_progress(step: str, progress: int):
                job_store.update(job_id, step=step, progress=progress)

            result = run_pipeline(pipeline_input, on_progress=_on_progress)
            job_store.update(job_id, status="completed", progress=100, result=result)
        except ShowtimeError as e:
            job_store.update(job_id, status="failed", error=str(e))
        except Exception as e:
            job_store.update(job_id, status="failed", error=f"Unexpected error: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return JobStatus(
        job_id=job_id,
        status="processing",
        progress=0,
        step="Starting",
    )
