"""Timeline assembly — build an edit decision list from AI mappings.

Handles:
- Ordering clips by voiceover sentence sequence
- Smart sub-segment splitting when a segment is reused by multiple sentences
- "Show" clips during voiceover silences (plays actual video, not frozen frame)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

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
        clips.append(
            TimelineClip(
                order=order,
                source_start=vid_start,
                source_end=vid_end,
                speed_factor=mapping.speed_factor,
                audio_start=sentence.start,
                audio_end=sentence.end,
                freeze=mapping.freeze,
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

    Distributes the total usable video time proportionally across all
    timeline events (sentences + gaps), so gaps get actual video content
    instead of frozen frames.
    """
    # Collect all timeline events in order: sentences and gaps between them
    events: list[tuple[str, float]] = []  # (key, duration)

    for i, mapping in enumerate(sorted_mappings):
        sentence = sentences_by_id.get(mapping.sentence_id)
        if sentence is None:
            continue

        # Check for gap before this sentence
        if i > 0:
            prev_sentence = sentences_by_id[sorted_mappings[i - 1].sentence_id]
            gap_duration = sentence.start - prev_sentence.end
            if gap_duration >= _MIN_GAP_DURATION:
                events.append((f"gap_{i}", gap_duration))

        events.append((f"sent_{mapping.sentence_id}", sentence.duration))

    total_event_duration = sum(dur for _, dur in events)

    # Get total usable video from the segments referenced by mappings
    all_seg_ids = {m.segment_id for m in sorted_mappings}
    all_segs = sorted([segments_by_id[sid] for sid in all_seg_ids if sid in segments_by_id],
                      key=lambda s: s.start)

    if not all_segs:
        return {key: (0.0, 0.0) for key, _ in events}

    video_start = all_segs[0].start
    video_end = all_segs[-1].end
    total_video = video_end - video_start

    # Cap video to avoid using dead time beyond what makes sense
    usable_video = min(total_video, total_event_duration * 1.2)

    if total_event_duration <= 0:
        return {key: (video_start, video_start) for key, _ in events}

    # Distribute video proportionally across all events
    positions: dict[str, tuple[float, float]] = {}
    cursor = video_start

    for key, duration in events:
        proportion = duration / total_event_duration
        slice_duration = usable_video * proportion
        pos_start = cursor
        pos_end = min(cursor + slice_duration, video_end)
        positions[key] = (round(pos_start, 3), round(pos_end, 3))
        cursor = pos_end

    return positions
