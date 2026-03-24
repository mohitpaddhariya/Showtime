"""In-memory job storage for MVP. Replace with Redis/DB later."""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.models.schemas import PipelineResult


@dataclass
class Job:
    job_id: str
    status: str = "pending"  # pending | processing | completed | failed
    progress: int = 0
    step: str | None = None
    error: str | None = None
    video_path: Path | None = None
    audio_path: Path | None = None
    output_path: Path | None = None
    work_dir: Path | None = None
    result: PipelineResult | None = None
    created_at: float = field(default_factory=time.time)
    video_filename: str | None = None
    audio_filename: str | None = None


class JobStore:
    """Thread-safe in-memory job store with SSE event broadcasting."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # SSE subscribers: job_id -> list of asyncio.Queue
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def create(self, video_path: Path, audio_path: Path, output_path: Path, work_dir: Path) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            job_id=job_id,
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            work_dir=work_dir,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job and job.work_dir and job.work_dir.exists():
            shutil.rmtree(job.work_dir, ignore_errors=True)
        return job is not None

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)
        # Broadcast SSE event to subscribers
        self._broadcast(job_id)

    # ── SSE support ──────────────────────────────────────────

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            if queue in subs:
                subs.remove(queue)

    def _broadcast(self, job_id: str) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id, [])
        for queue in subs:
            try:
                queue.put_nowait(job_id)
            except asyncio.QueueFull:
                pass


# Singleton instance
job_store = JobStore()
