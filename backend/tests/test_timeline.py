"""Tests for the timeline assembly pipeline component."""

from pathlib import Path

import pytest

from app.core.exceptions import TimelineError
from app.models.domain import CaptionedSegment, MappingEntry, VoiceoverSentence
from app.services.timeline import assemble_timeline


# ── Shared fixtures ──────────────────────────────────────────────────


@pytest.fixture
def segments() -> list[CaptionedSegment]:
    return [
        CaptionedSegment(segment_id=1, start=0.0, end=5.0, description="Page A"),
        CaptionedSegment(segment_id=2, start=5.0, end=10.0, description="Page B"),
        CaptionedSegment(segment_id=3, start=10.0, end=15.0, description="Page C"),
    ]


@pytest.fixture
def sentences() -> list[VoiceoverSentence]:
    return [
        VoiceoverSentence(sentence_id=1, text="First sentence.", start=0.0, end=3.0),
        VoiceoverSentence(sentence_id=2, text="Second sentence.", start=3.0, end=7.0),
        VoiceoverSentence(sentence_id=3, text="Third sentence.", start=7.0, end=10.0),
    ]


VIDEO = Path("video.mp4")
AUDIO = Path("audio.wav")


# ── Core functionality ────────────────────────────────────────────────


class TestAssembleTimeline:
    def test_clip_count_matches_mappings(self, segments, sentences):
        mappings = [
            MappingEntry(sentence_id=1, segment_id=1),
            MappingEntry(sentence_id=2, segment_id=2),
            MappingEntry(sentence_id=3, segment_id=3),
        ]
        tl = assemble_timeline(mappings, segments, sentences, VIDEO, AUDIO)
        assert len(tl.clips) == 3

    def test_clips_ordered_by_sentence_id(self, segments, sentences):
        # Provide mappings in reverse order
        mappings = [
            MappingEntry(sentence_id=3, segment_id=3),
            MappingEntry(sentence_id=1, segment_id=1),
            MappingEntry(sentence_id=2, segment_id=2),
        ]
        tl = assemble_timeline(mappings, segments, sentences, VIDEO, AUDIO)
        # Clips should be ordered by sentence, not by input order
        assert tl.clips[0].audio_start == 0.0  # sentence 1
        assert tl.clips[1].audio_start == 3.0  # sentence 2
        assert tl.clips[2].audio_start == 7.0  # sentence 3

    def test_sequential_order_values(self, segments, sentences):
        mappings = [
            MappingEntry(sentence_id=1, segment_id=1),
            MappingEntry(sentence_id=2, segment_id=2),
        ]
        tl = assemble_timeline(mappings, segments, sentences[:2], VIDEO, AUDIO)
        assert [c.order for c in tl.clips] == [0, 1]

    def test_source_times_within_segment_range(self, segments, sentences):
        mappings = [
            MappingEntry(sentence_id=1, segment_id=2),  # maps to segment 2
        ]
        tl = assemble_timeline(mappings, segments, sentences[:1], VIDEO, AUDIO)
        # Video position is within segment 2's range (5.0-10.0)
        assert tl.clips[0].source_start >= 5.0
        assert tl.clips[0].source_end <= 10.0

    def test_audio_times_from_sentences(self, segments, sentences):
        mappings = [
            MappingEntry(sentence_id=2, segment_id=1),
        ]
        tl = assemble_timeline(mappings, segments, sentences[1:2], VIDEO, AUDIO)
        assert tl.clips[0].audio_start == 3.0  # sentence 2 start
        assert tl.clips[0].audio_end == 7.0  # sentence 2 end

    def test_speed_factor_preserved(self, segments, sentences):
        mappings = [
            MappingEntry(sentence_id=1, segment_id=1, speed_factor=2.0),
        ]
        tl = assemble_timeline(mappings, segments, sentences[:1], VIDEO, AUDIO)
        assert tl.clips[0].speed_factor == 2.0

    def test_total_duration_matches_audio(self, segments, sentences):
        mappings = [
            MappingEntry(sentence_id=1, segment_id=1),
            MappingEntry(sentence_id=2, segment_id=2),
            MappingEntry(sentence_id=3, segment_id=3),
        ]
        tl = assemble_timeline(mappings, segments, sentences, VIDEO, AUDIO)
        expected = sum(s.duration for s in sentences)
        assert tl.total_duration == pytest.approx(expected)

    def test_source_paths_stored(self, segments, sentences):
        mappings = [MappingEntry(sentence_id=1, segment_id=1)]
        tl = assemble_timeline(mappings, segments, sentences[:1], VIDEO, AUDIO)
        assert tl.source_video == VIDEO
        assert tl.source_audio == AUDIO

    def test_segment_reused_by_multiple_sentences(self, segments, sentences):
        # Both sentences mapped to segment 1 (0.0-5.0s)
        mappings = [
            MappingEntry(sentence_id=1, segment_id=1),
            MappingEntry(sentence_id=2, segment_id=1),
        ]
        tl = assemble_timeline(mappings, segments, sentences[:2], VIDEO, AUDIO)
        assert len(tl.clips) == 2
        # Segment should be split: clip 1 gets first portion, clip 2 gets rest
        assert tl.clips[0].source_start == 0.0
        assert tl.clips[0].source_end < 5.0  # not the full segment
        assert tl.clips[1].source_start == tl.clips[0].source_end  # contiguous
        assert tl.clips[1].source_end == 5.0


# ── Edge cases ────────────────────────────────────────────────────────


class TestAssembleTimelineEdgeCases:
    def test_empty_mappings(self, segments, sentences):
        tl = assemble_timeline([], segments, sentences, VIDEO, AUDIO)
        assert len(tl.clips) == 0
        assert tl.total_duration == 0.0

    def test_single_mapping(self, segments, sentences):
        mappings = [MappingEntry(sentence_id=1, segment_id=1)]
        tl = assemble_timeline(mappings, segments, sentences[:1], VIDEO, AUDIO)
        assert len(tl.clips) == 1
        assert tl.clips[0].order == 0


# ── Gap insertion ─────────────────────────────────────────────────────


class TestGapInsertion:
    def test_gap_inserted_between_sentences_with_pause(self, segments):
        """Sentences with a gap > 0.15s should produce a gap clip."""
        sentences_with_gap = [
            VoiceoverSentence(sentence_id=1, text="First.", start=0.0, end=3.0),
            VoiceoverSentence(sentence_id=2, text="Second.", start=4.0, end=7.0),  # 1s gap
        ]
        mappings = [
            MappingEntry(sentence_id=1, segment_id=1),
            MappingEntry(sentence_id=2, segment_id=2),
        ]
        tl = assemble_timeline(mappings, segments, sentences_with_gap, VIDEO, AUDIO)
        # Should have 3 clips: content, gap, content
        assert len(tl.clips) == 3
        assert tl.clips[1].is_gap is True
        assert tl.clips[1].audio_start == pytest.approx(3.0)
        assert tl.clips[1].audio_end == pytest.approx(4.0)

    def test_no_gap_for_back_to_back_sentences(self, segments, sentences):
        """Contiguous sentences (no pause) should not produce gap clips."""
        mappings = [
            MappingEntry(sentence_id=1, segment_id=1),
            MappingEntry(sentence_id=2, segment_id=2),
            MappingEntry(sentence_id=3, segment_id=3),
        ]
        tl = assemble_timeline(mappings, segments, sentences, VIDEO, AUDIO)
        assert len(tl.clips) == 3
        assert all(not c.is_gap for c in tl.clips)

    def test_gap_duration_preserved(self, segments):
        sentences = [
            VoiceoverSentence(sentence_id=1, text="A.", start=0.0, end=2.0),
            VoiceoverSentence(sentence_id=2, text="B.", start=2.8, end=5.0),  # 0.8s gap
        ]
        mappings = [
            MappingEntry(sentence_id=1, segment_id=1),
            MappingEntry(sentence_id=2, segment_id=2),
        ]
        tl = assemble_timeline(mappings, segments, sentences, VIDEO, AUDIO)
        gap_clip = tl.clips[1]
        assert gap_clip.is_gap is True
        assert gap_clip.rendered_duration == pytest.approx(0.8)

    def test_total_duration_includes_gaps(self, segments):
        sentences = [
            VoiceoverSentence(sentence_id=1, text="A.", start=0.0, end=2.0),
            VoiceoverSentence(sentence_id=2, text="B.", start=3.0, end=5.0),  # 1s gap
        ]
        mappings = [
            MappingEntry(sentence_id=1, segment_id=1),
            MappingEntry(sentence_id=2, segment_id=2),
        ]
        tl = assemble_timeline(mappings, segments, sentences, VIDEO, AUDIO)
        # Total = 2.0 (sent1) + 1.0 (gap) + 2.0 (sent2) = 5.0
        assert tl.total_duration == pytest.approx(5.0)


# ── Error handling ────────────────────────────────────────────────────


class TestAssembleTimelineErrors:
    def test_invalid_segment_id_raises(self, segments, sentences):
        mappings = [MappingEntry(sentence_id=1, segment_id=999)]
        with pytest.raises(TimelineError, match="nonexistent segment_id"):
            assemble_timeline(mappings, segments, sentences[:1], VIDEO, AUDIO)

    def test_invalid_sentence_id_raises(self, segments, sentences):
        mappings = [MappingEntry(sentence_id=999, segment_id=1)]
        with pytest.raises(TimelineError, match="nonexistent sentence_id"):
            assemble_timeline(mappings, segments, sentences, VIDEO, AUDIO)
