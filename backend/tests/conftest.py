"""Shared pytest fixtures for the Showtime test suite."""

import struct
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("showtime_test")


@pytest.fixture(scope="session")
def sample_video(work_dir: Path) -> Path:
    """Generate a 3-second synthetic video with 3 distinct color scenes (1s each)."""
    path = work_dir / "sample.mp4"
    fps = 10
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    # 3 scenes: red, green, blue — each 1 second (10 frames)
    colors = [(0, 0, 200), (0, 200, 0), (200, 0, 0)]  # BGR
    for color in colors:
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        for _ in range(fps):
            writer.write(frame)

    writer.release()
    return path


@pytest.fixture(scope="session")
def sample_audio(work_dir: Path) -> Path:
    """Generate a 3-second sine-wave WAV file at 16kHz."""
    path = work_dir / "sample.wav"
    sample_rate = 16000
    duration = 3.0
    freq = 440.0
    n_samples = int(sample_rate * duration)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = b"".join(
            struct.pack("<h", int(32767 * np.sin(2 * np.pi * freq * i / sample_rate)))
            for i in range(n_samples)
        )
        wf.writeframes(samples)

    return path


@pytest.fixture
def keyframes_dir(tmp_path: Path) -> Path:
    """A temporary directory for keyframe output."""
    d = tmp_path / "keyframes"
    d.mkdir()
    return d
