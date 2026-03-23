"""In-memory job storage for MVP. Replace with Redis/DB later."""

from __future__ import annotations

import threading
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


class JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

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

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)


# Singleton instance
job_store = JobStore()
