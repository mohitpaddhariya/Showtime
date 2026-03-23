"""Tests for shared Pydantic models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.domain import (
    CaptionedSegment,
    MappingEntry,
    Timeline,
    TimelineClip,
    VideoSegment,
    VoiceoverSentence,
)
from app.models.schemas import PipelineInput, PipelineResult


class TestVideoSegment:
    def test_duration(self):
        seg = VideoSegment(segment_id=1, start=1.0, end=3.5)
        assert seg.duration == 2.5

    def test_defaults(self):
        seg = VideoSegment(segment_id=1, start=0.0, end=5.0)
        assert seg.is_idle is False
        assert seg.keyframe_path is None

    def test_with_keyframe(self, tmp_path: Path):
        kf = tmp_path / "frame.png"
        seg = VideoSegment(segment_id=1, start=0.0, end=1.0, keyframe_path=kf)
        assert seg.keyframe_path == kf


class TestCaptionedSegment:
    def test_description_default(self):
        seg = CaptionedSegment(segment_id=1, start=0.0, end=5.0)
        assert seg.description == ""

    def test_duration(self):
        seg = CaptionedSegment(segment_id=1, start=2.0, end=7.0, description="test")
        assert seg.duration == 5.0


class TestVoiceoverSentence:
    def test_duration(self):
        s = VoiceoverSentence(sentence_id=1, text="Hello world.", start=0.0, end=2.5)
        assert s.duration == 2.5

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            VoiceoverSentence(sentence_id=1, start=0.0, end=1.0)  # missing text


class TestMappingEntry:
    def test_defaults(self):
        m = MappingEntry(sentence_id=1, segment_id=1)
        assert m.speed_factor == 1.0

    def test_speed_factor_too_high(self):
        with pytest.raises(ValidationError):
            MappingEntry(sentence_id=1, segment_id=1, speed_factor=10.0)

    def test_speed_factor_too_low(self):
        with pytest.raises(ValidationError):
            MappingEntry(sentence_id=1, segment_id=1, speed_factor=0.1)

    def test_valid_speed_factor_bounds(self):
        m_low = MappingEntry(sentence_id=1, segment_id=1, speed_factor=0.25)
        m_high = MappingEntry(sentence_id=1, segment_id=1, speed_factor=4.0)
        assert m_low.speed_factor == 0.25
        assert m_high.speed_factor == 4.0


class TestTimelineClip:
    def test_rendered_duration(self):
        clip = TimelineClip(
            order=0,
            source_start=0.0,
            source_end=5.0,
            speed_factor=1.5,
            audio_start=0.0,
            audio_end=3.0,
        )
        assert clip.rendered_duration == 3.0


class TestTimeline:
    def test_total_duration(self):
        clips = [
            TimelineClip(
                order=0, source_start=0, source_end=5, audio_start=0, audio_end=3
            ),
            TimelineClip(
                order=1, source_start=5, source_end=10, audio_start=3, audio_end=7
            ),
        ]
        tl = Timeline(
            clips=clips, source_video=Path("v.mp4"), source_audio=Path("a.wav")
        )
        assert tl.total_duration == 7.0

    def test_empty_timeline(self):
        tl = Timeline(
            clips=[], source_video=Path("v.mp4"), source_audio=Path("a.wav")
        )
        assert tl.total_duration == 0.0


class TestPipelineModels:
    def test_pipeline_input(self, tmp_path: Path):
        inp = PipelineInput(
            video_path=Path("v.mp4"),
            audio_path=Path("a.wav"),
            output_path=Path("out.mp4"),
            work_dir=tmp_path,
        )
        assert inp.video_path == Path("v.mp4")

    def test_pipeline_result(self):
        result = PipelineResult(
            output_path=Path("out.mp4"),
            duration=30.0,
            segments_detected=5,
            sentences_detected=8,
            clips_rendered=8,
        )
        assert result.clips_rendered == 8
