"""Tests for the CLI pipeline orchestration."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.core.exceptions import ShowtimeError
from app.models.domain import (
    CaptionedSegment,
    MappingEntry,
    Timeline,
    TimelineClip,
    VideoSegment,
    VoiceoverSentence,
)
from app.models.schemas import PipelineInput, PipelineResult
from app.services.pipeline import run_pipeline

runner = CliRunner()


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_pipeline(mocker):
    """Mock all 6 pipeline functions and return the mocks."""
    segments = [
        VideoSegment(segment_id=1, start=0.0, end=5.0, keyframe_path=Path("/tmp/kf.png")),
    ]
    captioned = [
        CaptionedSegment(segment_id=1, start=0.0, end=5.0, description="Test page"),
    ]
    sentences = [
        VoiceoverSentence(sentence_id=1, text="This is a test.", start=0.0, end=3.0),
    ]
    mappings = [
        MappingEntry(sentence_id=1, segment_id=1, speed_factor=1.0),
    ]
    timeline = Timeline(
        clips=[
            TimelineClip(
                order=0, source_start=0.0, source_end=5.0,
                speed_factor=1.0, audio_start=0.0, audio_end=3.0,
            ),
        ],
        source_video=Path("v.mp4"),
        source_audio=Path("a.wav"),
    )

    mocks = {
        "detect_scenes": mocker.patch(
            "app.services.pipeline.detect_scenes", return_value=segments
        ),
        "caption_segments": mocker.patch(
            "app.services.pipeline.caption_segments", return_value=captioned
        ),
        "transcribe_audio": mocker.patch(
            "app.services.pipeline.transcribe_audio", return_value=sentences
        ),
        "map_sentences_to_segments": mocker.patch(
            "app.services.pipeline.map_sentences_to_segments", return_value=mappings
        ),
        "assemble_timeline": mocker.patch(
            "app.services.pipeline.assemble_timeline", return_value=timeline
        ),
        "render": mocker.patch(
            "app.services.pipeline.render", return_value=Path("out.mp4")
        ),
    }
    return mocks


# ── Tests for run_pipeline ────────────────────────────────────────────


class TestRunPipeline:
    def test_returns_pipeline_result(self, mock_pipeline, tmp_path):
        pipeline_input = PipelineInput(
            video_path=Path("v.mp4"),
            audio_path=Path("a.wav"),
            output_path=Path("out.mp4"),
            work_dir=tmp_path,
        )
        result = run_pipeline(pipeline_input)
        assert isinstance(result, PipelineResult)
        assert result.segments_detected == 1
        assert result.sentences_detected == 1
        assert result.clips_rendered == 1
        assert result.duration == 3.0

    def test_calls_all_pipeline_steps(self, mock_pipeline, tmp_path):
        pipeline_input = PipelineInput(
            video_path=Path("v.mp4"),
            audio_path=Path("a.wav"),
            output_path=Path("out.mp4"),
            work_dir=tmp_path,
        )
        run_pipeline(pipeline_input)

        mock_pipeline["detect_scenes"].assert_called_once()
        mock_pipeline["caption_segments"].assert_called_once()
        mock_pipeline["transcribe_audio"].assert_called_once()
        mock_pipeline["map_sentences_to_segments"].assert_called_once()
        mock_pipeline["assemble_timeline"].assert_called_once()
        mock_pipeline["render"].assert_called_once()

    def test_pipeline_step_order(self, mock_pipeline, tmp_path):
        """Verify steps are called in the correct order via side effects."""
        call_order = []

        for name, mock in mock_pipeline.items():
            original_return = mock.return_value
            mock.side_effect = lambda *a, _name=name, _ret=original_return, **kw: (
                call_order.append(_name) or _ret
            )

        pipeline_input = PipelineInput(
            video_path=Path("v.mp4"),
            audio_path=Path("a.wav"),
            output_path=Path("out.mp4"),
            work_dir=tmp_path,
        )
        run_pipeline(pipeline_input)

        assert call_order == [
            "detect_scenes",
            "caption_segments",
            "transcribe_audio",
            "map_sentences_to_segments",
            "assemble_timeline",
            "render",
        ]


class TestIdleFiltering:
    def test_idle_segments_filtered_before_captioning(self, mocker, tmp_path):
        """Pipeline should remove is_idle=True segments before caption/mapping."""
        segments_with_idle = [
            VideoSegment(segment_id=1, start=0.0, end=5.0, is_idle=False),
            VideoSegment(segment_id=2, start=5.0, end=15.0, is_idle=True),  # idle
            VideoSegment(segment_id=3, start=15.0, end=20.0, is_idle=False),
        ]
        captioned = [
            CaptionedSegment(segment_id=1, start=0.0, end=5.0, description="Page A"),
            CaptionedSegment(segment_id=3, start=15.0, end=20.0, description="Page B"),
        ]
        sentences = [
            VoiceoverSentence(sentence_id=1, text="Test.", start=0.0, end=3.0),
        ]
        mappings = [MappingEntry(sentence_id=1, segment_id=1, speed_factor=1.0)]
        timeline = Timeline(
            clips=[TimelineClip(order=0, source_start=0.0, source_end=5.0,
                                speed_factor=1.0, audio_start=0.0, audio_end=3.0)],
            source_video=Path("v.mp4"), source_audio=Path("a.wav"),
        )

        mocker.patch("app.services.pipeline.detect_scenes", return_value=segments_with_idle)
        caption_mock = mocker.patch("app.services.pipeline.caption_segments", return_value=captioned)
        mocker.patch("app.services.pipeline.transcribe_audio", return_value=sentences)
        mocker.patch("app.services.pipeline.map_sentences_to_segments", return_value=mappings)
        mocker.patch("app.services.pipeline.assemble_timeline", return_value=timeline)
        mocker.patch("app.services.pipeline.render", return_value=Path("out.mp4"))

        pipeline_input = PipelineInput(
            video_path=Path("v.mp4"), audio_path=Path("a.wav"),
            output_path=Path("out.mp4"), work_dir=tmp_path,
        )
        run_pipeline(pipeline_input)

        # caption_segments should receive only non-idle segments (2, not 3)
        called_segments = caption_mock.call_args[0][0]
        assert len(called_segments) == 2
        assert all(not s.is_idle for s in called_segments)


# ── Tests for CLI command ─────────────────────────────────────────────


class TestCLICommand:
    def test_missing_video_file(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        result = runner.invoke(app, [str(tmp_path / "nonexistent.mp4"), str(audio)])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_missing_audio_file(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        result = runner.invoke(app, [str(video), str(tmp_path / "nonexistent.wav")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_successful_run(self, mock_pipeline, tmp_path):
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.touch()
        audio.touch()
        output = tmp_path / "out.mp4"

        result = runner.invoke(app, [str(video), str(audio), "-o", str(output)])
        assert result.exit_code == 0
        assert "done" in result.output.lower()

    def test_pipeline_error_handled(self, mocker, tmp_path):
        mocker.patch(
            "app.services.pipeline.detect_scenes",
            side_effect=ShowtimeError("Something went wrong"),
        )
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.touch()
        audio.touch()

        result = runner.invoke(app, [str(video), str(audio)])
        assert result.exit_code == 1
        assert "pipeline error" in result.output.lower()

    def test_custom_output_path(self, mock_pipeline, tmp_path):
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.touch()
        audio.touch()
        output = tmp_path / "custom" / "result.mp4"

        result = runner.invoke(app, [str(video), str(audio), "-o", str(output)])
        assert result.exit_code == 0
