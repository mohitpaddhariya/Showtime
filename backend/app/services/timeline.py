"""Timeline assembly — build an edit decision list from AI mappings.

Handles:
- Ordering clips by voiceover sentence sequence
- Constrained video allocation: proportional splitting with minimum enforcement
  and speed-deviation minimization across shared segments
- "Show" clips during voiceover silences (plays actual video, not frozen)
- Audio-longer-than-video guard (slows/freezes, never drops sentences)
- Auto-freeze guard for zero-duration clips (segment exhausted)
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

# Minimum video slice per clip — clips shorter than this are unwatchable
_MIN_CLIP_VIDEO_DURATION = 1.0


def assemble_timeline(
    mappings: list[MappingEntry],
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    source_video: Path,
    source_audio: Path,
) -> Timeline:
    """Build an edit decision list from the AI mapping.

    For silent gaps between sentences, inserts "show" clips that play
    the actual video at 1x speed with silence — the viewer sees what's
    on screen during the narrator's pause.

    When voiceover audio is longer than source video, the video is
    slowed down or frozen — no sentences are dropped.
    """
    if not mappings:
        return Timeline(clips=[], source_video=source_video, source_audio=source_audio)

    segments_by_id = {s.segment_id: s for s in segments}
    sentences_by_id = {s.sentence_id: s for s in sentences}

    sorted_mappings = sorted(mappings, key=lambda m: m.sentence_id)

    # Compute video cursor positions for each sentence + gap using
    # constrained proportional allocation within each shared segment
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

        # Insert a "show" clip for voiceover gaps (natural pauses)
        if i > 0:
            prev_sentence = sentences_by_id.get(sorted_mappings[i - 1].sentence_id)
            if prev_sentence:
                gap_duration = sentence.start - prev_sentence.end
                if gap_duration >= _MIN_GAP_DURATION:
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

        # Auto-freeze zero-duration clips (segment video fully consumed)
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


def _compute_video_positions(
    sorted_mappings: list[MappingEntry],
    segments_by_id: dict[int, CaptionedSegment],
    sentences_by_id: dict[int, VoiceoverSentence],
) -> dict[str, tuple[float, float]]:
    """Compute video start/end for each sentence AND gap.

    Uses constrained proportional allocation:
    1. Group sentences by their mapped segment
    2. For each segment, split video time proportionally by audio duration
    3. Enforce minimum clip duration (when feasible without overflow)
    4. Redistribute to minimize speed deviation from ideal (seg_dur / total_audio)

    When minimum-duration enforcement would overflow a segment (audio >> video),
    automatically disables minimums. The video plays through at a consistent
    slow speed rather than restarting or freezing randomly.
    """
    if not sorted_mappings:
        return {}

    # ── Step 1: Group mappings by segment_id ─────────────────────────
    seg_sentence_groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for mapping in sorted_mappings:
        sentence = sentences_by_id.get(mapping.sentence_id)
        if sentence is None:
            continue
        seg_sentence_groups[mapping.segment_id].append(
            (mapping.sentence_id, sentence.duration)
        )

    # ── Step 2: Allocate sub-ranges within each segment ──────────────
    sentence_video_ranges: dict[int, tuple[float, float]] = {}

    for seg_id, sent_list in seg_sentence_groups.items():
        segment = segments_by_id.get(seg_id)
        if segment is None:
            continue

        seg_start = segment.start
        seg_dur = segment.end - segment.start
        total_audio = sum(dur for _, dur in sent_list)

        if total_audio <= 0 or seg_dur <= 0:
            for sent_id, _ in sent_list:
                sentence_video_ranges[sent_id] = (seg_start, segment.end)
            continue

        # If segment is too short to split meaningfully, give each sentence
        # the full range (renderer handles via speed or freeze)
        if len(sent_list) > 1 and seg_dur / len(sent_list) < _MIN_CLIP_VIDEO_DURATION:
            for sent_id, _ in sent_list:
                sentence_video_ranges[sent_id] = (seg_start, segment.end)
            continue

        # Check if proportional allocation with minimums would overflow
        min_total = sum(
            max(seg_dur * (dur / total_audio), min(_MIN_CLIP_VIDEO_DURATION, seg_dur))
            for _, dur in sent_list
        )
        use_minimums = min_total <= seg_dur * 1.05  # within 5% tolerance

        # Proportional allocation: each sentence gets video proportional
        # to its audio duration. This gives uniform speed across all clips
        # sharing the segment: speed = seg_dur / total_audio.
        allocations = []
        for _, aud_dur in sent_list:
            proportion = aud_dur / total_audio
            slice_dur = seg_dur * proportion
            if use_minimums:
                slice_dur = max(slice_dur, min(_MIN_CLIP_VIDEO_DURATION, seg_dur))
            allocations.append(slice_dur)

        # Normalize if allocations exceed segment duration (from minimum enforcement)
        alloc_total = sum(allocations)
        if alloc_total > seg_dur * 1.001:  # >0.1% over
            scale = seg_dur / alloc_total
            allocations = [a * scale for a in allocations]

        # Assign sequential video ranges within the segment
        cursor = seg_start
        for idx, (sent_id, _) in enumerate(sent_list):
            vid_start = cursor
            vid_end = min(cursor + allocations[idx], segment.end)
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
                    prev_vid_end = sentence_video_ranges.get(
                        sorted_mappings[i - 1].sentence_id, (0.0, 0.0)
                    )[1]
                    curr_vid_start = sentence_video_ranges.get(
                        mapping.sentence_id, (prev_vid_end, prev_vid_end)
                    )[0]

                    if curr_vid_start > prev_vid_end:
                        positions[f"gap_{i}"] = (round(prev_vid_end, 3), round(curr_vid_start, 3))
                    else:
                        # No gap in video — hold the last frame position briefly
                        positions[f"gap_{i}"] = (round(prev_vid_end, 3), round(prev_vid_end + 0.1, 3))

        # Content clip
        vid_range = sentence_video_ranges.get(
            mapping.sentence_id, (0.0, 0.0)
        )
        positions[f"sent_{mapping.sentence_id}"] = vid_range

    return positions
