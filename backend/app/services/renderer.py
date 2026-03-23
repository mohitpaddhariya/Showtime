"""FFmpeg-based video rendering from a Timeline.

Per-clip approach:
1. For each clip: trim video (speed-adjusted) + trim matching audio → muxed clip
2. Concatenate all clips (with optional crossfade) → final output

Each clip is a self-contained video+audio file, so concatenation always preserves audio.
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

_MIN_SPEED = 0.5
_MAX_SPEED = 3.0


def render(
    timeline: Timeline,
    output_path: Path,
    settings: Settings | None = None,
) -> Path:
    """Render the timeline to a final video file with synced audio.

    Each clip is rendered with its own video+audio, then all clips are concatenated.
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
            elif clip.freeze:
                _render_freeze_clip(clip, timeline, clip_path, src_props, settings)
            else:
                _render_content_clip(clip, timeline, clip_path, src_props, settings)
            clip_paths.append(clip_path)
            clip_durations.append(clip.rendered_duration)

        # Pass 2: Concatenate
        if len(clip_paths) == 1:
            _copy_file(clip_paths[0], output_path)
        else:
            _concatenate_simple(clip_paths, output_path, work_dir, settings)

    if not output_path.exists():
        raise RenderError(f"Render completed but output file not found: {output_path}")

    _validate_output(output_path)
    return output_path


# ── Clip rendering ────────────────────────────────────────────────────


def _render_content_clip(
    clip: TimelineClip,
    timeline: Timeline,
    output_path: Path,
    src_props: dict,
    settings: Settings,
) -> None:
    """Render one clip: speed-adjusted video + trimmed audio, muxed together."""
    video_duration = clip.source_end - clip.source_start
    audio_duration = clip.rendered_duration

    if audio_duration > 0 and video_duration > 0:
        effective_speed = video_duration / audio_duration
    else:
        effective_speed = 1.0

    # Handle extreme speeds
    filters = []
    if effective_speed > _MAX_SPEED:
        usable = audio_duration * _MAX_SPEED
        trim_start = clip.source_start + (video_duration - usable) / 2
        trim_duration = usable
        effective_speed = _MAX_SPEED
    elif effective_speed < _MIN_SPEED:
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
    filters.append(f"fps={src_props['fps']:.2f}")

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
        # Video filter
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
    """Render a freeze clip: hold a keyframe still while voiceover audio plays.

    Used when the LLM decides the viewer needs to read/absorb what's on screen.
    Video = single frame from the segment's midpoint, looped for the sentence duration.
    Audio = trimmed from voiceover at the sentence's timestamps.
    """
    audio_duration = clip.rendered_duration
    freeze_time = (clip.source_start + clip.source_end) / 2  # midpoint of segment
    fps = src_props["fps"]

    cmd = [
        "ffmpeg", "-y",
        # Video: grab one frame and loop it
        "-ss", f"{freeze_time:.3f}",
        "-i", str(timeline.source_video),
        # Audio: trim voiceover for this sentence
        "-ss", f"{clip.audio_start:.3f}",
        "-t", f"{audio_duration:.3f}",
        "-i", str(timeline.source_audio),
        # Video filter: loop single frame for exact duration
        "-filter:v", f"loop=loop=-1:size=1:start=0,fps={fps:.2f},setpts=N/FR/TB",
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
    """Render a 'show' clip: actual video playing at speed + silence.

    During voiceover pauses, the viewer should see the screen recording
    continuing to play (not a frozen frame). Audio is silent.
    """
    gap_duration = clip.rendered_duration
    video_duration = clip.source_end - clip.source_start

    # Calculate speed to match gap duration
    if gap_duration > 0 and video_duration > 0:
        effective_speed = video_duration / gap_duration
        effective_speed = max(0.1, min(5.0, effective_speed))
    else:
        effective_speed = 1.0

    video_filter = f"setpts=PTS/{effective_speed:.4f},fps={src_props['fps']:.2f}"

    cmd = [
        "ffmpeg", "-y",
        # Video: actual recording, speed-adjusted
        "-ss", f"{clip.source_start:.3f}",
        "-t", f"{video_duration:.3f}",
        "-i", str(timeline.source_video),
        # Audio: silence
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
    """Concatenate clips using concat demuxer."""
    concat_list = work_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", settings.video_codec,
        "-preset", settings.output_preset,
        "-crf", str(settings.crf),
        "-c:a", settings.audio_codec,
        "-ar", "44100",
        "-ac", "2",
        str(output_path),
    ]

    _run_ffmpeg(cmd, "concatenate clips")


# ── Utilities ─────────────────────────────────────────────────────────


def _copy_file(src: Path, dst: Path) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dst)]
    _run_ffmpeg(cmd, "copy single clip")


def _probe_video(video_path: Path) -> dict:
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
