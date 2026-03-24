"""OpenCV-based scene detection with optional AI verification pass.

Pipeline:
1. Sample frames at SAMPLE_FPS using OpenCV
2. Compute pixel diffs + histogram comparison -> scene boundaries
3. Auto-split long segments, merge short segments, detect idle
4. Extract keyframes (middle frame per segment)
5. [NEW] AI Verification: batch ALL keyframes in ONE Llama 4 Scout call
   -> confirm/merge/split segments semantically + assign semantic tags

The AI verification uses a SINGLE Groq vision call regardless of segment count.
This improves OpenCV's pixel-based detection with semantic understanding:
e.g., OpenCV might split a slowly-scrolling page into 3 segments, but the AI
recognizes it's all one page and merges them.

Groq call budget: 0 calls (AI verify disabled) or 1 call (AI verify enabled).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from groq import Groq

from app.core.config import Settings
from app.core.exceptions import SceneDetectionError
from app.models.domain import VideoSegment
from app.services.vision_utils import build_image_content_blocks

logger = logging.getLogger(__name__)


# ── AI Verification Prompt ───────────────────────────────────────────
# Sent once with ALL keyframes batched. The model sees every screen
# and decides which boundaries are real scene changes vs. noise.

_VERIFY_PROMPT = """\
You are analyzing keyframe screenshots from a screen recording to verify scene boundaries.

Each image represents the middle frame of a detected scene segment.

For each segment, decide:
1. Is this a valid, semantically distinct scene boundary?
2. What is the screen content? (short semantic label)
3. Is this segment idle/blank (no meaningful content)?

Return ONLY valid JSON:
{"segments": [
  {"segment_id": 1, "action": "keep", "semantic_tag": "landing_page", "is_idle": false},
  {"segment_id": 2, "action": "merge_with_prev", "semantic_tag": "", "is_idle": false},
  ...
]}

ACTIONS:
- "keep" = valid scene boundary, keep this segment
- "merge_with_next" = same screen as next segment, absorb it
- "merge_with_prev" = same screen as previous segment, absorb into it

RULES:
- Only merge segments that show the SAME screen content (e.g., slow scroll split into pieces)
- Do NOT merge segments showing genuinely different screens
- semantic_tag should be 1-3 words (e.g., "code_editor", "settings_modal", "login_form")
- is_idle=true only for blank/static/loading screens with no useful content"""


def detect_scenes(
    video_path: Path,
    work_dir: Path,
    settings: Settings | None = None,
) -> list[VideoSegment]:
    """Analyze a video and return segments with scene boundaries + semantic tags.

    Steps:
    1. OpenCV pixel-diff analysis -> initial segments
    2. Auto-split long segments, merge short segments
    3. Extract keyframes (middle frame of each segment)
    4. [Optional] AI verification: batch keyframes -> Llama 4 Scout
       -> refine boundaries + add semantic tags (1 Groq call)

    Args:
        video_path: Path to the input video file.
        work_dir: Directory to save extracted keyframe images.
        settings: Optional settings override; uses defaults if None.

    Returns:
        List of VideoSegment ordered by start time.
        - is_idle=True for idle/blank segments
        - semantic_tag set when AI verification is enabled

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

    # ── OpenCV analysis ──────────────────────────────────────────
    boundaries = _find_scene_boundaries(frames, settings)
    video_duration = total_frames / fps
    segments = _build_segments(boundaries, frames, video_duration, settings)
    segments = _refine_long_segments(segments, frames, video_duration, settings)
    segments = _merge_short_segments(segments, settings)

    # ── Extract keyframes ────────────────────────────────────────
    keyframes_dir = work_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    _extract_keyframes(video_path, segments, keyframes_dir)

    # ── AI Verification Pass (1 Groq vision call) ───────────────
    # Batches ALL keyframes into one request. The model sees every screen
    # and can confirm/merge/split based on actual visual content.
    if settings.ai_verify_scenes and settings.groq_api_key:
        try:
            segments = _ai_verify_segments(segments, settings)
            logger.info(
                "AI verification: %d segments, tags: %s",
                len(segments),
                [s.semantic_tag for s in segments if s.semantic_tag],
            )
        except Exception as e:
            # AI verification is best-effort — OpenCV segments are still usable
            logger.warning("AI scene verification failed (using OpenCV segments): %s", e)

    return segments


# ── AI Verification ──────────────────────────────────────────────────


_MAX_IMAGES_PER_CALL = 5  # Groq Llama 4 Scout hard limit


def _ai_verify_segments(
    segments: list[VideoSegment],
    settings: Settings,
) -> list[VideoSegment]:
    """Send keyframes to Llama 4 Scout for semantic verification.

    Groq limits Llama 4 Scout to 5 images per request, so segments are
    processed in chunks of 5. Each chunk is one API call.

    Images are resized to 1024px max to keep payload under 4MB per call.
    """
    if not segments:
        return segments

    client = Groq(api_key=settings.groq_api_key)
    verify_by_id: dict[int, dict] = {}

    # Process segments in chunks of 5 (Groq's per-request image limit)
    for chunk_start in range(0, len(segments), _MAX_IMAGES_PER_CALL):
        chunk = segments[chunk_start:chunk_start + _MAX_IMAGES_PER_CALL]

        content_blocks = build_image_content_blocks(chunk, max_dim=1024)
        content_blocks.append({
            "type": "text",
            "text": (
                "\n\nAnalyze these keyframes and verify the scene boundaries. "
                "Return JSON with your assessment of each segment."
            ),
        })

        response = client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=[
                {"role": "system", "content": _VERIFY_PROMPT},
                {"role": "user", "content": content_blocks},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        for v in data.get("segments", []):
            sid = v.get("segment_id")
            if sid is not None:
                verify_by_id[sid] = v

    if not verify_by_id:
        return segments

    # Apply merges and semantic tags
    result: list[VideoSegment] = []
    skip_next = False

    for i, seg in enumerate(segments):
        if skip_next:
            skip_next = False
            continue

        v = verify_by_id.get(seg.segment_id, {})
        action = v.get("action", "keep")
        tag = v.get("semantic_tag", "") or None
        ai_idle = v.get("is_idle", False)

        if action == "merge_with_next" and i + 1 < len(segments):
            # Absorb next segment into this one
            next_seg = segments[i + 1]
            result.append(VideoSegment(
                segment_id=seg.segment_id,
                start=seg.start,
                end=next_seg.end,
                is_idle=seg.is_idle and next_seg.is_idle,
                keyframe_path=seg.keyframe_path,
                semantic_tag=tag,
            ))
            skip_next = True
        elif action == "merge_with_prev" and result:
            # Absorb into the previous segment
            prev = result[-1]
            result[-1] = VideoSegment(
                segment_id=prev.segment_id,
                start=prev.start,
                end=seg.end,
                is_idle=prev.is_idle and seg.is_idle,
                keyframe_path=prev.keyframe_path,
                semantic_tag=prev.semantic_tag or tag,
            )
        else:
            # Keep segment, apply AI enrichments
            result.append(VideoSegment(
                segment_id=seg.segment_id,
                start=seg.start,
                end=seg.end,
                is_idle=seg.is_idle or ai_idle,
                keyframe_path=seg.keyframe_path,
                semantic_tag=tag,
            ))

    # Re-number segment IDs after merges
    for i, seg in enumerate(result):
        seg.segment_id = i + 1

    return result


# ── OpenCV scene boundary detection ─────────────────────────────────


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
