"""Tests for the renderer pipeline component."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import RenderError
from app.models.domain import Timeline, TimelineClip
from app.services.renderer import build_ffmpeg_clip_args, render, _MIN_SPEED, _MAX_SPEED


def _mock_subprocess(mocker):
    mock_run = mocker.patch("app.services.renderer.subprocess.run")

    def _side_effect(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[0] == "ffprobe":
            if "-show_streams" in cmd:
                result.stdout = json.dumps({
                    "streams": [{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1"}]
                })
            else:
                result.stdout = json.dumps({
                    "format": {"duration": "10.0"},
                    "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                })
        return result

    mock_run.side_effect = _side_effect
    mocker.patch.object(Path, "exists", return_value=True)
    return mock_run


@pytest.fixture
def simple_timeline(tmp_path):
    v, a = tmp_path / "src.mp4", tmp_path / "src.wav"
    v.touch(); a.touch()
    return Timeline(
        clips=[TimelineClip(order=0, source_start=0.0, source_end=5.0, audio_start=0.0, audio_end=3.0)],
        source_video=v, source_audio=a,
    )


@pytest.fixture
def multi_clip_timeline(tmp_path):
    v, a = tmp_path / "src.mp4", tmp_path / "src.wav"
    v.touch(); a.touch()
    return Timeline(
        clips=[
            TimelineClip(order=0, source_start=0.0, source_end=5.0, audio_start=0.0, audio_end=3.0),
            TimelineClip(order=1, source_start=5.0, source_end=10.0, audio_start=3.0, audio_end=7.0),
        ],
        source_video=v, source_audio=a,
    )


class TestBuildFfmpegClipArgs:
    def test_has_two_inputs(self):
        clip = TimelineClip(order=0, source_start=0.0, source_end=5.0, audio_start=0.0, audio_end=3.0)
        args = build_ffmpeg_clip_args(clip, Path("v.mp4"), Path("a.wav"), Path("out.mp4"), Settings())
        assert sum(1 for a in args if a == "-i") == 2

    def test_maps_video_and_audio(self):
        clip = TimelineClip(order=0, source_start=0.0, source_end=5.0, audio_start=0.0, audio_end=3.0)
        args = build_ffmpeg_clip_args(clip, Path("v.mp4"), Path("a.wav"), Path("out.mp4"), Settings())
        assert "0:v:0" in args
        assert "1:a:0" in args

    def test_speed_clamped_high(self):
        clip = TimelineClip(order=0, source_start=0.0, source_end=20.0, audio_start=0.0, audio_end=2.0)
        args = build_ffmpeg_clip_args(clip, Path("v.mp4"), Path("a.wav"), Path("out.mp4"), Settings())
        filter_idx = args.index("-filter:v")
        speed = float(args[filter_idx + 1].split("/")[1])
        assert speed == _MAX_SPEED

    def test_speed_clamped_low(self):
        clip = TimelineClip(order=0, source_start=0.0, source_end=1.0, audio_start=0.0, audio_end=10.0)
        args = build_ffmpeg_clip_args(clip, Path("v.mp4"), Path("a.wav"), Path("out.mp4"), Settings())
        filter_idx = args.index("-filter:v")
        speed = float(args[filter_idx + 1].split("/")[1])
        assert speed == _MIN_SPEED

    def test_normalizes_audio(self):
        clip = TimelineClip(order=0, source_start=0.0, source_end=5.0, audio_start=0.0, audio_end=3.0)
        args = build_ffmpeg_clip_args(clip, Path("v.mp4"), Path("a.wav"), Path("out.mp4"), Settings())
        assert "44100" in args
        assert "2" in args  # stereo


class TestRenderUnit:
    def test_calls_ffmpeg(self, mocker, simple_timeline, tmp_path):
        mock_run = _mock_subprocess(mocker)
        render(simple_timeline, tmp_path / "output.mp4")
        assert mock_run.called

    def test_single_clip_has_audio(self, mocker, simple_timeline, tmp_path):
        mock_run = _mock_subprocess(mocker)
        render(simple_timeline, tmp_path / "output.mp4")
        # Clip render should have two -i inputs (video + audio)
        for call in mock_run.call_args_list:
            cmd = call[0][0]
            if cmd[0] == "ffmpeg" and any("setpts" in str(a) for a in cmd):
                assert sum(1 for a in cmd if a == "-i") == 2
                return
        pytest.fail("No clip render with two inputs found")

    def test_multi_clip_concats(self, mocker, multi_clip_timeline, tmp_path):
        mock_run = _mock_subprocess(mocker)
        render(multi_clip_timeline, tmp_path / "output.mp4")
        all_cmds = " ".join(" ".join(str(a) for a in c[0][0]) for c in mock_run.call_args_list)
        assert "concat" in all_cmds


class TestGapClips:
    def test_gap_has_silence(self, mocker, tmp_path):
        mock_run = _mock_subprocess(mocker)
        v, a = tmp_path / "src.mp4", tmp_path / "src.wav"
        v.touch(); a.touch()
        tl = Timeline(
            clips=[
                TimelineClip(order=0, source_start=0.0, source_end=5.0, audio_start=0.0, audio_end=3.0),
                TimelineClip(order=1, source_start=4.9, source_end=5.0, audio_start=3.0, audio_end=3.5, is_gap=True),
                TimelineClip(order=2, source_start=5.0, source_end=10.0, audio_start=3.5, audio_end=7.0),
            ],
            source_video=v, source_audio=a,
        )
        render(tl, tmp_path / "output.mp4")
        all_cmds = " ".join(" ".join(str(a) for a in c[0][0]) for c in mock_run.call_args_list)
        assert "anullsrc" in all_cmds


class TestRenderErrors:
    def test_empty_timeline(self, tmp_path):
        tl = Timeline(clips=[], source_video=tmp_path / "v.mp4", source_audio=tmp_path / "a.wav")
        with pytest.raises(RenderError, match="empty timeline"):
            render(tl, tmp_path / "out.mp4")

    def test_ffmpeg_not_found(self, mocker, simple_timeline, tmp_path):
        mocker.patch("app.services.renderer.subprocess.run", side_effect=FileNotFoundError())
        with pytest.raises(RenderError, match="FFmpeg not found"):
            render(simple_timeline, tmp_path / "out.mp4")

    def test_ffmpeg_failure(self, mocker, simple_timeline, tmp_path):
        m = MagicMock(); m.returncode = 1; m.stderr = "Error"
        mocker.patch("app.services.renderer.subprocess.run", return_value=m)
        with pytest.raises(RenderError, match="FFmpeg failed"):
            render(simple_timeline, tmp_path / "out.mp4")


@pytest.mark.integration
class TestRenderIntegration:
    def test_renders_valid_mp4(self, sample_video, sample_audio, tmp_path):
        tl = Timeline(
            clips=[TimelineClip(order=0, source_start=0.0, source_end=3.0, audio_start=0.0, audio_end=3.0)],
            source_video=sample_video, source_audio=sample_audio,
        )
        output = tmp_path / "output.mp4"
        assert render(tl, output).exists()
