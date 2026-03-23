"""Tests for FastAPI endpoints."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import Job, JobStore, job_store
from app.main import app
from app.models.schemas import PipelineResult

client = TestClient(app)


# ── Health check ──────────────────────────────────────────────────────


class TestHealthCheck:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ── Upload endpoint ───────────────────────────────────────────────────


class TestUpload:
    def test_upload_success(self, tmp_path):
        video_content = b"fake video content"
        audio_content = b"fake audio content"

        response = client.post(
            "/api/v1/upload",
            files={
                "video": ("demo.mp4", io.BytesIO(video_content), "video/mp4"),
                "audio": ("narration.wav", io.BytesIO(audio_content), "audio/wav"),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 12

    def test_upload_invalid_video_format(self):
        response = client.post(
            "/api/v1/upload",
            files={
                "video": ("demo.txt", io.BytesIO(b"not a video"), "text/plain"),
                "audio": ("narration.wav", io.BytesIO(b"audio"), "audio/wav"),
            },
        )
        assert response.status_code == 400
        assert "unsupported video" in response.json()["detail"].lower()

    def test_upload_invalid_audio_format(self):
        response = client.post(
            "/api/v1/upload",
            files={
                "video": ("demo.mp4", io.BytesIO(b"video"), "video/mp4"),
                "audio": ("narration.pdf", io.BytesIO(b"not audio"), "application/pdf"),
            },
        )
        assert response.status_code == 400
        assert "unsupported audio" in response.json()["detail"].lower()

    def test_upload_creates_job_in_store(self):
        response = client.post(
            "/api/v1/upload",
            files={
                "video": ("demo.mp4", io.BytesIO(b"video"), "video/mp4"),
                "audio": ("narration.wav", io.BytesIO(b"audio"), "audio/wav"),
            },
        )
        job_id = response.json()["job_id"]
        job = job_store.get(job_id)
        assert job is not None
        assert job.status == "pending"


# ── Status endpoint ───────────────────────────────────────────────────


class TestStatus:
    def test_status_not_found(self):
        response = client.get("/api/v1/status/nonexistent")
        assert response.status_code == 404

    def test_status_pending(self):
        # Create a job via upload
        response = client.post(
            "/api/v1/upload",
            files={
                "video": ("demo.mp4", io.BytesIO(b"video"), "video/mp4"),
                "audio": ("narration.wav", io.BytesIO(b"audio"), "audio/wav"),
            },
        )
        job_id = response.json()["job_id"]

        status_response = client.get(f"/api/v1/status/{job_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["status"] == "pending"
        assert data["progress"] == 0


# ── Process endpoint ──────────────────────────────────────────────────


class TestProcess:
    def test_process_not_found(self):
        response = client.post("/api/v1/process/nonexistent")
        assert response.status_code == 404

    def test_process_starts(self, mocker):
        # Mock the pipeline so it doesn't actually run
        mocker.patch("app.api.v1.process.run_pipeline")
        mocker.patch("app.api.v1.process.threading.Thread")

        response = client.post(
            "/api/v1/upload",
            files={
                "video": ("demo.mp4", io.BytesIO(b"video"), "video/mp4"),
                "audio": ("narration.wav", io.BytesIO(b"audio"), "audio/wav"),
            },
        )
        job_id = response.json()["job_id"]

        process_response = client.post(f"/api/v1/process/{job_id}")
        assert process_response.status_code == 200
        assert process_response.json()["status"] == "processing"


# ── Download endpoint ─────────────────────────────────────────────────


class TestDownload:
    def test_download_not_found(self):
        response = client.get("/api/v1/download/nonexistent")
        assert response.status_code == 404

    def test_download_not_completed(self):
        response = client.post(
            "/api/v1/upload",
            files={
                "video": ("demo.mp4", io.BytesIO(b"video"), "video/mp4"),
                "audio": ("narration.wav", io.BytesIO(b"audio"), "audio/wav"),
            },
        )
        job_id = response.json()["job_id"]

        download_response = client.get(f"/api/v1/download/{job_id}")
        assert download_response.status_code == 400

    def test_download_completed(self, tmp_path):
        # Manually create a completed job
        output = tmp_path / "output.mp4"
        output.write_bytes(b"fake mp4 content")

        result = PipelineResult(
            output_path=output, duration=10.0,
            segments_detected=3, sentences_detected=5, clips_rendered=5,
        )
        job = job_store.create(
            video_path=tmp_path / "v.mp4", audio_path=tmp_path / "a.wav",
            output_path=output, work_dir=tmp_path,
        )
        job_store.update(job.job_id, status="completed", result=result)

        response = client.get(f"/api/v1/download/{job.job_id}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"

    def test_status_includes_result_when_completed(self, tmp_path):
        """Status endpoint returns download_url + metadata when job is done."""
        output = tmp_path / "output.mp4"
        output.write_bytes(b"fake mp4")

        result = PipelineResult(
            output_path=output, duration=15.5,
            segments_detected=4, sentences_detected=6, clips_rendered=6,
        )
        job = job_store.create(
            video_path=tmp_path / "v.mp4", audio_path=tmp_path / "a.wav",
            output_path=output, work_dir=tmp_path,
        )
        job_store.update(job.job_id, status="completed", result=result)

        response = client.get(f"/api/v1/status/{job.job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["duration"] == 15.5
        assert data["clips_rendered"] == 6
        assert f"/download/{job.job_id}" in data["download_url"]


class TestUploadAndProcess:
    def test_one_step_upload_and_process(self, mocker):
        mocker.patch("app.api.v1.upload.run_pipeline")
        mocker.patch("app.api.v1.upload.threading.Thread")

        response = client.post(
            "/api/v1/upload-and-process",
            files={
                "video": ("demo.mp4", io.BytesIO(b"video"), "video/mp4"),
                "audio": ("narration.wav", io.BytesIO(b"audio"), "audio/wav"),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert "job_id" in data
