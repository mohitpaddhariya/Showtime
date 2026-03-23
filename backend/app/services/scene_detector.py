"""OpenCV-based scene detection, idle segment identification, and keyframe extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.core.exceptions import SceneDetectionError
from app.models.domain import VideoSegment


def detect_scenes(
    video_path: Path,
    work_dir: Path,
    settings: Settings | None = None,
) -> list[VideoSegment]:
    """Analyze a video and return a list of segments with scene boundaries.

    Args:
        video_path: Path to the input video file.
        work_dir: Directory to save extracted keyframe images.
        settings: Optional settings override; uses defaults if None.

    Returns:
        List of VideoSegment, one per detected scene, ordered by start time.
        Idle segments have is_idle=True.

    Raises:
        SceneDetectionError: If the video cannot be opened or read.
    """
    if settings is None:
        settings = Settings()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SceneDetectionError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise SceneDetectionError(f"Invalid FPS ({fps}) for video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(fps / settings.sample_fps))

    # Collect sampled frames with their timestamps
    frames: list[tuple[float, np.ndarray]] = []
    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append((timestamp, gray))

    cap.release()

    if not frames:
        raise SceneDetectionError(f"No frames could be read from: {video_path}")

    # Compute diffs and find scene boundaries
    boundaries = _find_scene_boundaries(frames, settings)

    # Build segments from boundaries
    video_duration = total_frames / fps
    segments = _build_segments(boundaries, frames, video_duration, settings)

    # Auto-split long segments with a lower threshold
    segments = _refine_long_segments(segments, frames, video_duration, settings)

    # Merge tiny segments into neighbors
    segments = _merge_short_segments(segments, settings)

    # Extract keyframes and save them
    keyframes_dir = work_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    _extract_keyframes(video_path, segments, keyframes_dir)

    return segments


def _find_scene_boundaries(
    frames: list[tuple[float, np.ndarray]],
    settings: Settings,
) -> list[int]:
    """Return indices into `frames` where scene changes occur.

    Uses two complementary methods:
    1. Mean absolute pixel difference (catches structural changes)
    2. Histogram comparison (catches color/layout changes that pixel diff misses)

    A scene change is detected if EITHER method exceeds its threshold.
    """
    boundaries: list[int] = [0]  # first frame always starts a scene

    for i in range(1, len(frames)):
        # Method 1: pixel diff
        diff = cv2.absdiff(frames[i][1], frames[i - 1][1])
        mean_diff = float(np.mean(diff))

        # Method 2: histogram comparison (catches page navigations, color changes)
        # Only triggers if there's also a meaningful pixel diff (> half the threshold)
        # This prevents histogram from overriding intentionally high thresholds
        hist_change = False
        if mean_diff > settings.scene_threshold * 0.5:
            hist_prev = cv2.calcHist([frames[i - 1][1]], [0], None, [64], [0, 256])
            hist_curr = cv2.calcHist([frames[i][1]], [0], None, [64], [0, 256])
            cv2.normalize(hist_prev, hist_prev)
            cv2.normalize(hist_curr, hist_curr)
            hist_corr = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
            # correlation < 0.85 means significant histogram shift
            hist_change = hist_corr < 0.85

        if mean_diff > settings.scene_threshold or hist_change:
            boundaries.append(i)

    return boundaries


def _build_segments(
    boundaries: list[int],
    frames: list[tuple[float, np.ndarray]],
    video_duration: float,
    settings: Settings,
) -> list[VideoSegment]:
    """Build VideoSegment objects from scene boundary indices."""
    segments: list[VideoSegment] = []

    for i, b_idx in enumerate(boundaries):
        start = frames[b_idx][0]

        # End is either the next boundary's timestamp or the video duration
        if i + 1 < len(boundaries):
            end = frames[boundaries[i + 1]][0]
        else:
            end = video_duration

        # Determine if this segment is idle by checking diffs within it
        is_idle = _check_idle(b_idx, boundaries, i, frames, settings)

        segments.append(
            VideoSegment(
                segment_id=i + 1,
                start=round(start, 3),
                end=round(end, 3),
                is_idle=is_idle,
            )
        )

    return segments


def _check_idle(
    b_idx: int,
    boundaries: list[int],
    boundary_pos: int,
    frames: list[tuple[float, np.ndarray]],
    settings: Settings,
) -> bool:
    """Check if a segment is idle (negligible visual change throughout).

    A segment is idle if:
    1. Its duration is at least `idle_min_duration` seconds.
    2. All consecutive frame diffs within the segment are below `idle_threshold`.
    """
    # Determine the frame range for this segment
    start_idx = b_idx
    if boundary_pos + 1 < len(boundaries):
        end_idx = boundaries[boundary_pos + 1]
    else:
        end_idx = len(frames)

    # Need at least 2 frames to check
    if end_idx - start_idx < 2:
        return False

    # Check duration
    segment_duration = frames[min(end_idx - 1, len(frames) - 1)][0] - frames[start_idx][0]
    if segment_duration < settings.idle_min_duration:
        return False

    # Check all diffs within the segment
    for j in range(start_idx + 1, end_idx):
        diff = cv2.absdiff(frames[j][1], frames[j - 1][1])
        mean_diff = float(np.mean(diff))
        if mean_diff > settings.idle_threshold:
            return False

    return True


def _merge_short_segments(
    segments: list[VideoSegment],
    settings: Settings,
) -> list[VideoSegment]:
    """Merge segments shorter than min_segment_duration into their neighbors.

    Tiny segments (cursor blinks, small UI changes) are noise — they get
    absorbed into the previous segment to keep the timeline clean.
    """
    min_dur = settings.min_segment_duration
    if not segments:
        return segments

    # Don't merge if video is short (< 10s) — every segment matters
    total_dur = segments[-1].end - segments[0].start
    if total_dur < 10.0:
        return segments

    merged: list[VideoSegment] = [segments[0]]

    for seg in segments[1:]:
        if seg.duration < min_dur:
            # Absorb into the previous segment
            prev = merged[-1]
            merged[-1] = VideoSegment(
                segment_id=prev.segment_id,
                start=prev.start,
                end=seg.end,
                is_idle=prev.is_idle and seg.is_idle,
            )
        else:
            merged.append(seg)

    # Also merge the first segment if it's too short
    if len(merged) > 1 and merged[0].duration < min_dur:
        merged[1] = VideoSegment(
            segment_id=merged[1].segment_id,
            start=merged[0].start,
            end=merged[1].end,
            is_idle=merged[0].is_idle and merged[1].is_idle,
        )
        merged.pop(0)

    # Re-number segment IDs
    for i, seg in enumerate(merged):
        seg.segment_id = i + 1

    return merged


def _refine_long_segments(
    segments: list[VideoSegment],
    frames: list[tuple[float, np.ndarray]],
    video_duration: float,
    settings: Settings,
) -> list[VideoSegment]:
    """Auto-split segments longer than max_segment_duration.

    Re-scans long segments with a lower threshold to find internal transitions
    that the primary scan missed (e.g., scrolling, tab switching, zoom changes).
    """
    max_dur = settings.max_segment_duration
    refine_thresh = settings.scene_refine_threshold
    refined: list[VideoSegment] = []
    seg_id = 1

    for seg in segments:
        if seg.duration <= max_dur:
            refined.append(VideoSegment(
                segment_id=seg_id, start=seg.start, end=seg.end,
                is_idle=seg.is_idle,
            ))
            seg_id += 1
            continue

        # Find frames within this segment's time range
        seg_frames = [
            (i, ts, gray) for i, (ts, gray) in enumerate(frames)
            if seg.start <= ts <= seg.end
        ]

        if len(seg_frames) < 2:
            refined.append(VideoSegment(
                segment_id=seg_id, start=seg.start, end=seg.end,
                is_idle=seg.is_idle,
            ))
            seg_id += 1
            continue

        # Find internal boundaries with lower threshold
        sub_boundaries = [0]  # start of sub-segment
        for j in range(1, len(seg_frames)):
            diff = cv2.absdiff(seg_frames[j][2], seg_frames[j - 1][2])
            mean_diff = float(np.mean(diff))
            if mean_diff > refine_thresh:
                sub_boundaries.append(j)

        # Build sub-segments
        for k, b_idx in enumerate(sub_boundaries):
            sub_start = seg_frames[b_idx][1]  # timestamp

            if k + 1 < len(sub_boundaries):
                sub_end = seg_frames[sub_boundaries[k + 1]][1]
            else:
                sub_end = seg.end

            refined.append(VideoSegment(
                segment_id=seg_id,
                start=round(sub_start, 3),
                end=round(sub_end, 3),
                is_idle=seg.is_idle,
            ))
            seg_id += 1

    return refined


def _extract_keyframes(
    video_path: Path,
    segments: list[VideoSegment],
    keyframes_dir: Path,
) -> None:
    """Extract the middle frame of each segment and save as PNG."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    for seg in segments:
        mid_time = (seg.start + seg.end) / 2
        mid_frame = int(mid_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if ret:
            kf_path = keyframes_dir / f"segment_{seg.segment_id:04d}.png"
            cv2.imwrite(str(kf_path), frame)
            seg.keyframe_path = kf_path

    cap.release()
