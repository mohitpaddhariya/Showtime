#!/usr/bin/env python3
"""End-to-end test with real video + audio files.

Usage:
    uv run python scripts/test_real_video.py path/to/video.mp4 path/to/audio.wav
    uv run python scripts/test_real_video.py path/to/video.mp4 path/to/audio.wav -o output.mp4
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.core.config import Settings
from app.core.exceptions import ShowtimeError
from app.models.schemas import PipelineInput
from app.services.scene_detector import detect_scenes
from app.services.frame_captioner import caption_segments
from app.services.audio_analyzer import transcribe_audio
from app.services.ai_mapper import map_sentences_to_segments
from app.services.timeline import assemble_timeline
from app.services.renderer import render

console = Console()


def main():
    if len(sys.argv) < 3:
        console.print("[red]Usage:[/] python scripts/test_real_video.py <video> <audio> [-o output.mp4]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    audio_path = Path(sys.argv[2])
    output_path = Path(sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "-o" else "output.mp4")

    if not video_path.exists():
        console.print(f"[red]Video not found:[/] {video_path}")
        sys.exit(1)
    if not audio_path.exists():
        console.print(f"[red]Audio not found:[/] {audio_path}")
        sys.exit(1)

    settings = Settings()
    timings: list[tuple[str, float]] = []

    console.print(f"\n[bold]Showtime — Real Video Test[/]")
    console.print(f"  Video:    {video_path}")
    console.print(f"  Audio:    {audio_path}")
    console.print(f"  Output:   {output_path}")
    console.print(f"  LLM:      {settings.llm_provider} ({settings.groq_model if settings.llm_provider == 'groq' else settings.ollama_model})")
    console.print(f"  Whisper:  {settings.whisper_provider} ({settings.groq_whisper_model if settings.whisper_provider == 'groq' else settings.whisper_model})")
    console.print()

    with tempfile.TemporaryDirectory(prefix="showtime_test_") as work_dir:
        work = Path(work_dir)

        try:
            # Step 1: Scene Detection
            console.print("[bold cyan]Step 1/6:[/] Detecting scenes...")
            t0 = time.perf_counter()
            segments = detect_scenes(video_path, work, settings)
            t1 = time.perf_counter()
            timings.append(("Scene Detection", t1 - t0))
            console.print(f"  → {len(segments)} segment(s) detected")

            # Step 2: Frame Captioning
            console.print("[bold cyan]Step 2/6:[/] Captioning keyframes...")
            t0 = time.perf_counter()
            captioned = caption_segments(segments)
            t1 = time.perf_counter()
            timings.append(("Frame Captioning", t1 - t0))
            for c in captioned:
                desc = c.description[:60] + "..." if len(c.description) > 60 else c.description
                console.print(f"  → Seg {c.segment_id} ({c.start:.1f}s-{c.end:.1f}s): {desc or '[no text]'}")

            # Step 3: Audio Transcription
            console.print("[bold cyan]Step 3/6:[/] Transcribing audio...")
            t0 = time.perf_counter()
            sentences = transcribe_audio(audio_path, settings)
            t1 = time.perf_counter()
            timings.append(("Transcription", t1 - t0))
            for s in sentences:
                console.print(f"  → Sen {s.sentence_id} ({s.start:.1f}s-{s.end:.1f}s): \"{s.text}\"")

            # Step 4: AI Mapping + Refinement
            console.print("[bold cyan]Step 4/6:[/] Mapping sentences to segments (+ refinement)...")
            t0 = time.perf_counter()
            mappings = map_sentences_to_segments(captioned, sentences, settings)
            t1 = time.perf_counter()
            timings.append(("AI Mapping + Refine", t1 - t0))
            for m in mappings:
                mode = "[blue]FREEZE[/]" if m.freeze else "[green]PLAY[/]"
                sen = next(s for s in sentences if s.sentence_id == m.sentence_id)
                console.print(f"  → Sen {m.sentence_id} → Seg {m.segment_id} [{mode}] \"{sen.text[:60]}\"")

            # Step 5: Timeline Assembly
            console.print("[bold cyan]Step 5/6:[/] Assembling timeline...")
            t0 = time.perf_counter()
            timeline = assemble_timeline(mappings, captioned, sentences, video_path, audio_path)
            t1 = time.perf_counter()
            timings.append(("Timeline Assembly", t1 - t0))

            # Show actual clip pacing (after sub-segment splitting)
            for c in timeline.clips:
                vdur = c.source_end - c.source_start
                adur = c.rendered_duration
                speed = vdur / adur if adur > 0 else 0
                label = "GAP " if c.is_gap else ("FRZ " if c.freeze else "PLAY")
                console.print(f"  [{label}] vid={c.source_start:.1f}-{c.source_end:.1f}s ({vdur:.1f}s) | aud={c.audio_start:.1f}-{c.audio_end:.1f}s ({adur:.1f}s) | {speed:.1f}x")
            console.print(f"  → {len(timeline.clips)} clip(s), {timeline.total_duration:.1f}s total")

            # Step 6: Rendering
            console.print("[bold cyan]Step 6/6:[/] Rendering final video...")
            t0 = time.perf_counter()
            render(timeline, output_path, settings)
            t1 = time.perf_counter()
            timings.append(("Rendering", t1 - t0))

        except ShowtimeError as e:
            console.print(f"\n[red]Pipeline error:[/] {e}")
            sys.exit(1)

    # Print timing summary
    total = sum(t for _, t in timings)
    console.print()

    table = Table(title="Timing Summary")
    table.add_column("Step", style="cyan")
    table.add_column("Time", justify="right", style="green")
    table.add_column("% of Total", justify="right")

    for step, duration in timings:
        pct = (duration / total * 100) if total > 0 else 0
        table.add_row(step, f"{duration:.2f}s", f"{pct:.0f}%")

    table.add_row("[bold]Total[/]", f"[bold]{total:.2f}s[/]", "[bold]100%[/]")
    console.print(table)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        console.print(f"\n[bold green]Done![/] Output: {output_path} ({size_mb:.1f} MB)")
    else:
        console.print(f"\n[bold red]Output file not created![/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
