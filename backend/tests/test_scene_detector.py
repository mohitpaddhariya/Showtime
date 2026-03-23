"""Tests for the scene detector pipeline component."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import SceneDetectionError
from app.services.scene_detector import (
    _build_segments,
    _check_idle,
    _find_scene_boundaries,
    detect_scenes,
)


# ── Integration tests (use real OpenCV with synthetic video) ──────────


class TestDetectScenesIntegration:
    """Integration tests using the synthetic sample_video fixture."""

    def test_detects_three_scenes(self, sample_video: Path, work_dir: Path):
        out_dir = work_dir / "scene_int_1"
        out_dir.mkdir(exist_ok=True)
        segments = detect_scenes(sample_video, out_dir)
        assert len(segments) == 3

    def test_no_idle_segments(self, sample_video: Path, work_dir: Path):
        out_dir = work_dir / "scene_int_2"
        out_dir.mkdir(exist_ok=True)
        segments = detect_scenes(sample_video, out_dir)
        assert all(not s.is_idle for s in segments)

    def test_keyframes_saved(self, sample_video: Path, work_dir: Path):
        out_dir = work_dir / "scene_int_3"
        out_dir.mkdir(exist_ok=True)
        segments = detect_scenes(sample_video, out_dir)
        for seg in segments:
            assert seg.keyframe_path is not None
            assert seg.keyframe_path.exists()
            assert seg.keyframe_path.suffix == ".png"

    def test_segments_cover_full_duration(self, sample_video: Path, work_dir: Path):
        out_dir = work_dir / "scene_int_4"
        out_dir.mkdir(exist_ok=True)
        segments = detect_scenes(sample_video, out_dir)
        assert segments[0].start == pytest.approx(0.0, abs=0.5)
        # 3 second video at 10fps = 30 frames
        assert segments[-1].end == pytest.approx(3.0, abs=0.5)

    def test_segments_are_contiguous(self, sample_video: Path, work_dir: Path):
        out_dir = work_dir / "scene_int_5"
        out_dir.mkdir(exist_ok=True)
        segments = detect_scenes(sample_video, out_dir)
        for i in range(1, len(segments)):
            assert segments[i].start == segments[i - 1].end

    def test_segment_ids_sequential(self, sample_video: Path, work_dir: Path):
        out_dir = work_dir / "scene_int_6"
        out_dir.mkdir(exist_ok=True)
        segments = detect_scenes(sample_video, out_dir)
        ids = [s.segment_id for s in segments]
        assert ids == [1, 2, 3]


# ── Error handling ────────────────────────────────────────────────────


class TestDetectScenesErrors:
    def test_nonexistent_video_raises(self, tmp_path: Path):
        with pytest.raises(SceneDetectionError, match="Cannot open video"):
            detect_scenes(Path("/nonexistent/video.mp4"), tmp_path)

    def test_invalid_file_raises(self, tmp_path: Path):
        bad_file = tmp_path / "bad.mp4"
        bad_file.write_text("not a video")
        with pytest.raises(SceneDetectionError, match="Cannot open video"):
            detect_scenes(bad_file, tmp_path)


# ── Unit tests for internal functions ─────────────────────────────────


class TestFindSceneBoundaries:
    def _make_frames(self, colors: list[int], repeats: int = 5) -> list[tuple[float, np.ndarray]]:
        """Create synthetic grayscale frames with given intensity values."""
        frames = []
        t = 0.0
        for color in colors:
            frame = np.full((100, 100), color, dtype=np.uint8)
            for _ in range(repeats):
                frames.append((t, frame.copy()))
                t += 0.5
        return frames

    def test_two_distinct_scenes(self):
        # 5 dark frames, then 5 bright frames
        frames = self._make_frames([10, 200], repeats=5)
        settings = Settings(scene_threshold=30.0)
        boundaries = _find_scene_boundaries(frames, settings)
        # Should detect boundary at index 0 (always) and where brightness jumps
        assert len(boundaries) == 2
        assert boundaries[0] == 0
        assert boundaries[1] == 5

    def test_three_distinct_scenes(self):
        frames = self._make_frames([10, 200, 50], repeats=5)
        settings = Settings(scene_threshold=30.0)
        boundaries = _find_scene_boundaries(frames, settings)
        assert len(boundaries) == 3

    def test_uniform_video_single_scene(self):
        frames = self._make_frames([128], repeats=20)
        settings = Settings(scene_threshold=30.0)
        boundaries = _find_scene_boundaries(frames, settings)
        assert len(boundaries) == 1  # only the initial boundary

    def test_high_threshold_merges_scenes(self):
        # Two scenes with small diff (10 vs 50) — should NOT trigger at high threshold
        frames = self._make_frames([10, 50], repeats=5)
        settings = Settings(scene_threshold=100.0)
        boundaries = _find_scene_boundaries(frames, settings)
        assert len(boundaries) == 1


class TestCheckIdle:
    def test_identical_frames_are_idle(self):
        frames = [(i * 0.5, np.full((100, 100), 128, dtype=np.uint8)) for i in range(10)]
        settings = Settings(idle_threshold=2.0, idle_min_duration=1.0)
        # Single segment spanning all frames
        assert _check_idle(0, [0], 0, frames, settings) is True

    def test_changing_frames_not_idle(self):
        frames = []
        for i in range(10):
            val = 50 + i * 20  # increasing brightness
            frames.append((i * 0.5, np.full((100, 100), val, dtype=np.uint8)))
        settings = Settings(idle_threshold=2.0, idle_min_duration=1.0)
        assert _check_idle(0, [0], 0, frames, settings) is False

    def test_short_segment_not_idle(self):
        # Only 2 frames at 0.5s apart = 0.5s duration, below idle_min_duration
        frames = [(i * 0.5, np.full((100, 100), 128, dtype=np.uint8)) for i in range(2)]
        settings = Settings(idle_threshold=2.0, idle_min_duration=1.0)
        assert _check_idle(0, [0], 0, frames, settings) is False


class TestBuildSegments:
    def test_single_segment(self):
        frames = [(i * 0.5, np.full((100, 100), 128, dtype=np.uint8)) for i in range(6)]
        settings = Settings()
        segments = _build_segments([0], frames, 3.0, settings)
        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 3.0
        assert segments[0].segment_id == 1

    def test_two_segments(self):
        frames = [(i * 0.5, np.full((100, 100), 128, dtype=np.uint8)) for i in range(6)]
        settings = Settings()
        segments = _build_segments([0, 3], frames, 3.0, settings)
        assert len(segments) == 2
        assert segments[0].end == frames[3][0]
        assert segments[1].start == frames[3][0]
        assert segments[1].end == 3.0


# ── Idle detection integration with synthetic video ───────────────────


class TestIdleDetectionIntegration:
    def test_idle_video_detected(self, tmp_path: Path):
        """A video with all identical frames should produce an idle segment."""
        video_path = tmp_path / "idle.mp4"
        fps = 10
        width, height = 160, 120
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        # 3 seconds of identical frames
        frame = np.full((height, width, 3), (128, 128, 128), dtype=np.uint8)
        for _ in range(fps * 3):
            writer.write(frame)
        writer.release()

        out_dir = tmp_path / "idle_out"
        out_dir.mkdir()
        segments = detect_scenes(video_path, out_dir)
        assert len(segments) == 1
        assert segments[0].is_idle is True
