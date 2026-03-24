"""SSE endpoint — real-time progress streaming for frontend."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from app.api.jobs import job_store

router = APIRouter()


@router.get("/events/{job_id}")
async def stream_events(job_id: str):
    """Server-Sent Events stream for real-time job progress.

    Frontend usage:
        const es = new EventSource('/api/v1/events/<job_id>');
        es.onmessage = (e) => { const data = JSON.parse(e.data); ... };
        es.addEventListener('complete', (e) => { es.close(); });
        es.addEventListener('error', (e) => { es.close(); });
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    queue = job_store.subscribe(job_id)

    async def _event_generator():
        try:
            # Send current state immediately
            yield _format_sse(job_id)

            while True:
                try:
                    await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    continue

                current = job_store.get(job_id)
                if not current:
                    yield _format_sse(job_id, event="error", extra={"error": "Job deleted"})
                    break

                if current.status == "completed":
                    yield _format_sse(job_id, event="complete")
                    break
                elif current.status == "failed":
                    yield _format_sse(job_id, event="error")
                    break
                else:
                    yield _format_sse(job_id)
        finally:
            job_store.unsubscribe(job_id, queue)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse(job_id: str, event: str | None = None, extra: dict | None = None) -> str:
    """Format a Server-Sent Event message."""
    job = job_store.get(job_id)
    data = extra or {}
    if job:
        data.update({
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "step": job.step,
            "error": job.error,
        })
        if job.status == "completed" and job.result:
            data["download_url"] = f"/api/v1/download/{job_id}"
            data["duration"] = job.result.duration
            data["segments_detected"] = job.result.segments_detected
            data["sentences_detected"] = job.result.sentences_detected
            data["clips_rendered"] = job.result.clips_rendered

    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data)}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)
