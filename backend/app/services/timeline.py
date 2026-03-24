"""Timeline assembly — build an edit decision list from AI mappings.

Handles:
- Ordering clips by voiceover sentence sequence
- Smart sub-segment splitting when a segment is reused by multiple sentences
- "Show" clips during voiceover silences (plays actual video, not frozen frame)
- Audio-longer-than-video capping (drops sentences beyond video end)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

from app.core.exceptions import TimelineError
from app.models.domain import (
    CaptionedSegment,
    MappingEntry,
    Timeline,
    TimelineClip,
    VoiceoverSentence,
)

# Minimum gap between sentences to insert a show/silence clip (seconds)
_MIN_GAP_DURATION = 0.15


def assemble_timeline(
    mappings: list[MappingEntry],
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    source_video: Path,
    source_audio: Path,
) -> Timeline:
    """Build an edit decision list from the AI mapping.

    When the voiceover has silent gaps between sentences, inserts "show" clips
    that play the actual video at 1x speed with silence — so the viewer can
    see what's on screen during the narrator's pause.

    When the voiceover audio is longer than the source video, the video is
    slowed down or frozen to wait for the audio, then resumes — no sentences
    are dropped.
    """
    if not mappings:
        return Timeline(clips=[], source_video=source_video, source_audio=source_audio)

    segments_by_id = {s.segment_id: s for s in segments}
    sentences_by_id = {s.sentence_id: s for s in sentences}

    sorted_mappings = sorted(mappings, key=lambda m: m.sentence_id)

    # Compute the video cursor positions for each sentence + gap
    # The video is distributed across ALL time (sentences + gaps), not just sentences
    video_positions = _compute_video_positions(sorted_mappings, segments_by_id, sentences_by_id)

    clips: list[TimelineClip] = []
    order = 0

    for i, mapping in enumerate(sorted_mappings):
        segment = segments_by_id.get(mapping.segment_id)
        if segment is None:
            raise TimelineError(f"Mapping references nonexistent segment_id {mapping.segment_id}")

        sentence = sentences_by_id.get(mapping.sentence_id)
        if sentence is None:
            raise TimelineError(f"Mapping references nonexistent sentence_id {mapping.sentence_id}")

        # Insert a "show" clip if there's a gap before this sentence
        if i > 0:
            prev_sentence = sentences_by_id.get(sorted_mappings[i - 1].sentence_id)
            if prev_sentence:
                gap_duration = sentence.start - prev_sentence.end
                if gap_duration >= _MIN_GAP_DURATION:
                    # Play actual video during the gap (not a frozen frame)
                    gap_vid_start, gap_vid_end = video_positions[f"gap_{i}"]
                    clips.append(
                        TimelineClip(
                            order=order,
                            source_start=gap_vid_start,
                            source_end=gap_vid_end,
                            speed_factor=1.0,
                            audio_start=prev_sentence.end,
                            audio_end=sentence.start,
                            is_gap=True,
                        )
                    )
                    order += 1

        # Content clip
        vid_start, vid_end = video_positions[f"sent_{mapping.sentence_id}"]

        # Auto-freeze zero-duration clips (video exhausted for this segment)
        force_freeze = vid_start >= vid_end and not mapping.freeze
        if force_freeze:
            logger.debug(
                "Auto-freezing sentence %d (video exhausted at %.3fs)",
                mapping.sentence_id, vid_start,
            )

        clips.append(
            TimelineClip(
                order=order,
                source_start=vid_start,
                source_end=vid_end,
                speed_factor=mapping.speed_factor,
                audio_start=sentence.start,
                audio_end=sentence.end,
                freeze=mapping.freeze or force_freeze,
            )
        )
        order += 1

    return Timeline(clips=clips, source_video=source_video, source_audio=source_audio)


_MIN_CLIP_VIDEO_DURATION = 1.0  # seconds — clips shorter than this are unwatchable


def _compute_video_positions(
    sorted_mappings: list[MappingEntry],
    segments_by_id: dict[int, CaptionedSegment],
    sentences_by_id: dict[int, VoiceoverSentence],
) -> dict[str, tuple[float, float]]:
    """Compute video start/end for each sentence AND gap.

    RESPECTS the AI mapping: each sentence gets video from the segment
    it was mapped to. When multiple sentences share a segment, the segment's
    time range is split proportionally among them.

    Enforces a minimum video duration per clip to prevent sub-second slices
    that render as unwatchable blurs.
    """
    if not sorted_mappings:
        return {}

    # ── Step 1: Group mappings by segment_id ─────────────────────────
    # Track which sentences share each segment and their audio durations
    seg_sentence_groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for mapping in sorted_mappings:
        sentence = sentences_by_id.get(mapping.sentence_id)
        if sentence is None:
            continue
        seg_sentence_groups[mapping.segment_id].append(
            (mapping.sentence_id, sentence.duration)
        )

    # ── Step 2: Allocate sub-ranges within each segment ──────────────
    # For each segment, divide its video time among the sentences mapped to it
    # Enforces minimum video duration to prevent tiny unwatchable slices
    sentence_video_ranges: dict[int, tuple[float, float]] = {}

    for seg_id, sent_list in seg_sentence_groups.items():
        segment = segments_by_id.get(seg_id)
        if segment is None:
            continue

        seg_start = segment.start
        seg_dur = segment.end - segment.start
        total_audio = sum(dur for _, dur in sent_list)

        if total_audio <= 0 or seg_dur <= 0:
            # All sentences get the full segment range
            for sent_id, _ in sent_list:
                sentence_video_ranges[sent_id] = (seg_start, segment.end)
            continue

        # If segment is too short to split meaningfully, give each sentence
        # the full segment range (renderer will handle via speed or freeze)
        if len(sent_list) > 1 and seg_dur / len(sent_list) < _MIN_CLIP_VIDEO_DURATION:
            for sent_id, _ in sent_list:
                sentence_video_ranges[sent_id] = (seg_start, segment.end)
            continue

        # Check if proportional allocation with minimums would overflow.
        # If so, skip the minimum enforcement and use pure proportional split.
        # This makes each sentence advance sequentially through the segment
        # at a consistent slow speed (e.g. 0.3x) — no restarts, no crazy speeds.
        min_total = sum(
            max(seg_dur * (dur / total_audio), min(_MIN_CLIP_VIDEO_DURATION, seg_dur))
            for _, dur in sent_list
        )
        use_minimums = min_total <= seg_dur * 1.05  # within 5% = safe to enforce mins

        # Distribute segment time proportionally by audio duration
        cursor = seg_start
        for sent_id, aud_dur in sent_list:
            proportion = aud_dur / total_audio
            slice_dur = seg_dur * proportion
            # Only enforce minimum when it won't cause overflow
            if use_minimums:
                slice_dur = max(slice_dur, min(_MIN_CLIP_VIDEO_DURATION, seg_dur))
            vid_start = cursor
            vid_end = min(cursor + slice_dur, segment.end)
            sentence_video_ranges[sent_id] = (round(vid_start, 3), round(vid_end, 3))
            cursor = vid_end

    # ── Step 3: Build positions dict (sentences + gaps) ──────────────
    positions: dict[str, tuple[float, float]] = {}

    for i, mapping in enumerate(sorted_mappings):
        sentence = sentences_by_id.get(mapping.sentence_id)
        if sentence is None:
            continue

        # Insert gap clip between sentences
        if i > 0:
            prev_sentence = sentences_by_id.get(sorted_mappings[i - 1].sentence_id)
            if prev_sentence:
                gap_duration = sentence.start - prev_sentence.end
                if gap_duration >= _MIN_GAP_DURATION:
                    # For gaps: use video between previous sentence's end and current sentence's start
                    prev_vid_end = sentence_video_ranges.get(
                        sorted_mappings[i - 1].sentence_id, (0.0, 0.0)
                    )[1]
                    curr_vid_start = sentence_video_ranges.get(
                        mapping.sentence_id, (prev_vid_end, prev_vid_end)
                    )[0]

                    # If there's video space between them, use it; otherwise hold the last position
                    if curr_vid_start > prev_vid_end:
                        positions[f"gap_{i}"] = (round(prev_vid_end, 3), round(curr_vid_start, 3))
                    else:
                        # No gap in video — hold the last frame position briefly
                        positions[f"gap_{i}"] = (round(prev_vid_end, 3), round(prev_vid_end + 0.1, 3))

        # Content clip: use the segment range assigned by AI mapping
        vid_range = sentence_video_ranges.get(
            mapping.sentence_id, (0.0, 0.0)
        )
        positions[f"sent_{mapping.sentence_id}"] = vid_range

    return positions
