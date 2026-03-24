"""FFmpeg-based video rendering from a Timeline.

Per-clip approach:
1. For each clip: trim video (speed-adjusted) + trim matching audio -> muxed clip
2. Concatenate all clips (stream copy, no re-encode) -> final output

Clip types:
- Content: speed-adjusted video + voiceover audio (speed clamped 0.5-2.5x)
- Freeze: single keyframe with Ken Burns effect (subtle zoom) + voiceover audio
- Gap: video at speed + generated silence (during narrator pauses)

Ken Burns effect on freeze clips (NEW):
When a clip is frozen (static keyframe), applying a very subtle slow zoom
(1.0x -> 1.03x over the clip duration) prevents the "dead screen" effect
and keeps the viewer engaged. This is configurable via settings.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import RenderError
from app.models.domain import Timeline, TimelineClip

logger = logging.getLogger(__name__)

# Fallback constants (used when settings not available, e.g. build_ffmpeg_clip_args)
_MIN_SPEED = 0.5
_MAX_SPEED = 2.5

# Speed above which we auto-freeze instead of playing choppy fast-forward
_AUTO_FREEZE_THRESHOLD = 2.5


def render(
    timeline: Timeline,
    output_path: Path,
    settings: Settings | None = None,
) -> Path:
    """Render the timeline to a final video file with synced audio.

    Each clip is rendered individually with its own video+audio, then all
    clips are concatenated via concat demuxer (stream copy, no re-encode).
    """
    if settings is None:
        settings = Settings()

    if not timeline.clips:
        raise RenderError("Cannot render an empty timeline")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    src_props = _probe_video(timeline.source_video)

    with tempfile.TemporaryDirectory(prefix="showtime_render_") as tmp_dir:
        work_dir = Path(tmp_dir)

        # Pass 1: Render each clip with synced video + audio
        clip_paths = []
        clip_durations = []
        for clip in timeline.clips:
            clip_path = work_dir / f"clip_{clip.order:04d}.mp4"
            if clip.is_gap:
                _render_gap_clip(clip, timeline, clip_path, src_props, settings)
            elif clip.freeze or _should_auto_freeze(clip, settings):
                _render_freeze_clip(clip, timeline, clip_path, src_props, settings)
            else:
                _render_content_clip(clip, timeline, clip_path, src_props, settings)
            clip_paths.append(clip_path)
            clip_durations.append(clip.rendered_duration)

        # Pass 2: Concatenate (stream copy — no re-encode)
        if len(clip_paths) == 1:
            _copy_file(clip_paths[0], output_path)
        else:
            _concatenate_simple(clip_paths, output_path, work_dir, settings)

    if not output_path.exists():
        raise RenderError(f"Render completed but output file not found: {output_path}")

    _validate_output(output_path)
    return output_path


def _should_auto_freeze(clip: TimelineClip, settings: Settings) -> bool:
    """Auto-freeze clips that would need extreme speed-up.

    When video-to-audio ratio exceeds the max speed threshold, the result
    is unwatchably choppy. Hold a keyframe still instead.
    """
    video_duration = clip.source_end - clip.source_start
    audio_duration = clip.rendered_duration
    if audio_duration <= 0 or video_duration <= 0:
        return False
    effective_speed = video_duration / audio_duration
    return effective_speed > settings.max_playback_speed


# ── Clip rendering ────────────────────────────────────────────────────


def _render_content_clip(
    clip: TimelineClip,
    timeline: Timeline,
    output_path: Path,
    src_props: dict,
    settings: Settings,
) -> None:
    """Render one clip: speed-adjusted video + trimmed audio, muxed together.

    Speed is clamped to [min_playback_speed, max_playback_speed].
    If speed exceeds max, the video is center-cropped in time to fit.
    If speed is below min, padding is added to fill the gap.
    """
    video_duration = clip.source_end - clip.source_start
    audio_duration = clip.rendered_duration

    max_speed = settings.max_playback_speed
    min_speed = settings.min_playback_speed

    if audio_duration > 0 and video_duration > 0:
        effective_speed = video_duration / audio_duration
    else:
        effective_speed = 1.0

    # Handle extreme speeds
    filters = []
    if effective_speed > max_speed:
        # Too fast — center-crop the video in time to reduce speed
        usable = audio_duration * max_speed
        trim_start = clip.source_start + (video_duration - usable) / 2
        trim_duration = usable
        effective_speed = max_speed
    elif effective_speed < min_speed:
        # Too slow — play what we have, pad the rest with frame cloning
        trim_start = clip.source_start
        trim_duration = video_duration
        adjusted = trim_duration / effective_speed
        pad = audio_duration - adjusted
        if pad > 0:
            filters.append(f"tpad=stop_mode=clone:stop_duration={pad:.3f}")
        effective_speed = max(0.1, effective_speed)
    else:
        trim_start = clip.source_start
        trim_duration = video_duration

    filters.insert(0, f"setpts=PTS/{effective_speed:.4f}")

    # Preserve source fps (capped at 60) for smoothness
    out_fps = min(src_props["fps"], 60.0)
    filters.append(f"fps={out_fps:.2f}")

    cmd = [
        "ffmpeg", "-y",
        # Input 0: video
        "-ss", f"{trim_start:.3f}",
        "-t", f"{trim_duration:.3f}",
        "-i", str(timeline.source_video),
        # Input 1: audio
        "-ss", f"{clip.audio_start:.3f}",
        "-t", f"{audio_duration:.3f}",
        "-i", str(timeline.source_audio),
        # Video filter chain
        "-filter:v", ",".join(filters),
        # Map video from input 0, audio from input 1
        "-map", "0:v:0",
        "-map", "1:a:0",
        # Encoding
        "-c:v", settings.video_codec,
        "-preset", settings.output_preset,
        "-crf", str(settings.crf),
        "-c:a", settings.audio_codec,
        "-ar", "44100",
        "-ac", "2",
        # Force output duration to match audio
        "-t", f"{audio_duration:.3f}",
        str(output_path),
    ]

    _run_ffmpeg(cmd, f"content clip {clip.order}")


def _render_freeze_clip(
    clip: TimelineClip,
    timeline: Timeline,
    output_path: Path,
    src_props: dict,
    settings: Settings,
) -> None:
    """Render a freeze clip with optional Ken Burns effect.

    Without Ken Burns: single frame looped for the clip duration (static).
    With Ken Burns: subtle slow zoom from 1.0x to ~1.03x, centered, which
    prevents the "dead screen" effect and keeps viewers engaged.

    Video = single frame from segment midpoint, zoomed/looped for duration.
    Audio = trimmed from voiceover at the sentence's timestamps.
    """
    audio_duration = clip.rendered_duration
    freeze_time = (clip.source_start + clip.source_end) / 2
    fps = src_props["fps"]
    width = src_props["width"]
    height = src_props["height"]

    # Build video filter chain
    use_ken_burns = settings.ken_burns_on_freeze and audio_duration > 0.5
    if use_ken_burns:
        # Ken Burns: subtle slow zoom centered on the frame
        # zoompan reads one frame and produces a zoom animation
        total_frames = max(int(audio_duration * fps), 1)
        max_zoom = settings.ken_burns_zoom
        # Calculate per-frame zoom increment to reach max_zoom over the clip
        zoom_step = (max_zoom - 1.0) / max(total_frames, 1)
        video_filter = (
            f"zoompan=z='min(zoom+{zoom_step:.6f},{max_zoom:.4f})"
            f"':d={total_frames}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={width}x{height}:fps={fps:.2f}"
        )
    else:
        # Simple frame loop (no Ken Burns)
        video_filter = f"loop=loop=-1:size=1:start=0,fps={fps:.2f},setpts=N/FR/TB"

    cmd = [
        "ffmpeg", "-y",
        # Video: grab one frame
        "-ss", f"{freeze_time:.3f}",
        "-i", str(timeline.source_video),
        # Audio: trim voiceover for this sentence
        "-ss", f"{clip.audio_start:.3f}",
        "-t", f"{audio_duration:.3f}",
        "-i", str(timeline.source_audio),
        # Video filter
        "-filter:v", video_filter,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", f"{audio_duration:.3f}",
        "-c:v", settings.video_codec,
        "-preset", settings.output_preset,
        "-crf", str(settings.crf),
        "-c:a", settings.audio_codec,
        "-ar", "44100",
        "-ac", "2",
        str(output_path),
    ]

    _run_ffmpeg(cmd, f"freeze clip {clip.order}")


def _render_gap_clip(
    clip: TimelineClip,
    timeline: Timeline,
    output_path: Path,
    src_props: dict,
    settings: Settings,
) -> None:
    """Render a gap clip: video + silence during voiceover pauses.

    When there's enough video for smooth playback, plays video at speed.
    When video is too short (speed < min), holds a clean still frame
    instead of playing extreme slow-motion that looks laggy/glitchy.
    """
    gap_duration = clip.rendered_duration
    video_duration = clip.source_end - clip.source_start

    max_speed = settings.max_playback_speed
    min_speed = settings.min_playback_speed

    if gap_duration > 0 and video_duration > 0:
        raw_speed = video_duration / gap_duration
    else:
        raw_speed = 0.0

    # When video is too short for smooth playback (e.g., 0.1s video over
    # 0.7s gap = 0.14x speed), hold a clean still frame + silence.
    # This prevents the "laggy" feel during natural speech pauses.
    if raw_speed < min_speed:
        freeze_time = clip.source_start
        fps = src_props["fps"]

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{freeze_time:.3f}",
            "-i", str(timeline.source_video),
            "-f", "lavfi",
            "-t", f"{gap_duration:.3f}",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-filter:v", f"loop=loop=-1:size=1:start=0,fps={fps:.2f},setpts=N/FR/TB",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", f"{gap_duration:.3f}",
            "-c:v", settings.video_codec,
            "-preset", settings.output_preset,
            "-crf", str(settings.crf),
            "-c:a", settings.audio_codec,
            "-ar", "44100",
            "-ac", "2",
            str(output_path),
        ]
        _run_ffmpeg(cmd, f"gap-freeze clip {clip.order}")
        return

    # Normal gap: enough video for smooth playback at clamped speed
    effective_speed = max(min_speed, min(max_speed, raw_speed))
    out_fps = min(src_props["fps"], 60.0)
    video_filter = f"setpts=PTS/{effective_speed:.4f},fps={out_fps:.2f}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip.source_start:.3f}",
        "-t", f"{video_duration:.3f}",
        "-i", str(timeline.source_video),
        "-f", "lavfi",
        "-t", f"{gap_duration:.3f}",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-filter:v", video_filter,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", f"{gap_duration:.3f}",
        "-c:v", settings.video_codec,
        "-preset", settings.output_preset,
        "-crf", str(settings.crf),
        "-c:a", settings.audio_codec,
        "-ar", "44100",
        "-ac", "2",
        str(output_path),
    ]

    _run_ffmpeg(cmd, f"show clip {clip.order}")


# ── Concatenation ─────────────────────────────────────────────────────


def _concatenate_simple(
    clip_paths: list[Path],
    output_path: Path,
    work_dir: Path,
    settings: Settings,
) -> None:
    """Concatenate clips using concat demuxer with stream copy (no re-encode)."""
    concat_list = work_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",  # stream copy — no re-encode since clips are already encoded
        str(output_path),
    ]

    _run_ffmpeg(cmd, "concatenate clips")


# ── Utilities ─────────────────────────────────────────────────────────


def _copy_file(src: Path, dst: Path) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dst)]
    _run_ffmpeg(cmd, "copy single clip")


def _probe_video(video_path: Path) -> dict:
    """Extract video properties (fps, width, height) via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", str(video_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                s = streams[0]
                fps_parts = s.get("r_frame_rate", "30/1").split("/")
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
                return {
                    "width": int(s.get("width", 1920)),
                    "height": int(s.get("height", 1080)),
                    "fps": min(fps, 60.0),
                }
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    return {"width": 1920, "height": 1080, "fps": 30.0}


def _validate_output(output_path: Path) -> None:
    """Validate the rendered output has both video and audio streams."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-show_entries", "stream=codec_type", "-print_format", "json",
             str(output_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        if duration < 0.5:
            raise RenderError(f"Output too short ({duration:.1f}s)")
        codec_types = {s.get("codec_type") for s in data.get("streams", [])}
        if "video" not in codec_types:
            raise RenderError("Output missing video stream")
        if "audio" not in codec_types:
            raise RenderError("Output missing audio stream")
        logger.info("Output validated: %.1fs", duration)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("ffprobe not available, skipping validation")


def _run_ffmpeg(cmd: list[str], description: str) -> None:
    """Execute an FFmpeg command with error handling."""
    logger.debug("FFmpeg [%s]: %s", description, " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise RenderError("FFmpeg not found. Install with: brew install ffmpeg")
    if result.returncode != 0:
        raise RenderError(
            f"FFmpeg failed [{description}] (exit {result.returncode}):\n"
            f"{result.stderr[-1000:] if result.stderr else 'no stderr'}"
        )


def build_ffmpeg_clip_args(
    clip: TimelineClip,
    source_video: Path,
    source_audio: Path,
    output_path: Path,
    settings: Settings,
) -> list[str]:
    """Build FFmpeg command args for a single synced clip. Exposed for testing."""
    video_duration = clip.source_end - clip.source_start
    audio_duration = clip.rendered_duration
    effective_speed = (video_duration / audio_duration) if (audio_duration > 0 and video_duration > 0) else 1.0
    effective_speed = max(_MIN_SPEED, min(_MAX_SPEED, effective_speed))

    return [
        "ffmpeg", "-y",
        "-ss", f"{clip.source_start:.3f}",
        "-t", f"{video_duration:.3f}",
        "-i", str(source_video),
        "-ss", f"{clip.audio_start:.3f}",
        "-t", f"{audio_duration:.3f}",
        "-i", str(source_audio),
        "-filter:v", f"setpts=PTS/{effective_speed:.4f}",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", settings.video_codec,
        "-preset", settings.output_preset,
        "-crf", str(settings.crf),
        "-c:a", settings.audio_codec,
        "-ar", "44100",
        "-ac", "2",
        "-t", f"{audio_duration:.3f}",
        str(output_path),
    ]
