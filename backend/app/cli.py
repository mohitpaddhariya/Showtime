"""Showtime CLI — wire all pipeline steps together."""

from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from rich.console import Console

from app.core.exceptions import ShowtimeError
from app.models.schemas import PipelineInput
from app.services.pipeline import run_pipeline

app = typer.Typer(help="Showtime — turn rough screen recordings into polished demo videos.")
console = Console()


@app.command()
def process(
    video: Path = typer.Argument(..., help="Path to screen recording (.mp4/.mov/.avi/.mkv/.webm)"),
    audio: Path = typer.Argument(..., help="Path to voiceover audio (.mp3/.wav/.m4a/.aac/.ogg/.flac)"),
    output: Path = typer.Option("output.mp4", "--output", "-o", help="Output file path"),
    work_dir: Path | None = typer.Option(None, "--work-dir", "-w", help="Working directory (default: temp)"),
) -> None:
    """Turn a rough screen recording + voiceover into a polished demo video."""
    if not video.exists():
        console.print(f"[red]Error:[/] Video file not found: {video}")
        raise typer.Exit(code=1)

    if not audio.exists():
        console.print(f"[red]Error:[/] Audio file not found: {audio}")
        raise typer.Exit(code=1)

    def _on_progress(step: str, progress: int) -> None:
        console.print(f"[bold cyan][{progress:3d}%][/] {step}...")

    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        _run(video, audio, output, work_dir, _on_progress)
    else:
        with tempfile.TemporaryDirectory(prefix="showtime_") as tmp:
            _run(video, audio, output, Path(tmp), _on_progress)


def _run(video: Path, audio: Path, output: Path, work_dir: Path, on_progress: callable) -> None:
    """Run the pipeline with error handling."""
    pipeline_input = PipelineInput(
        video_path=video, audio_path=audio,
        output_path=output, work_dir=work_dir,
    )

    try:
        result = run_pipeline(pipeline_input, on_progress=on_progress)
        console.print()
        console.print("[bold green]Done![/]")
        console.print(f"  Output:   {result.output_path}")
        console.print(f"  Duration: {result.duration:.1f}s")
        console.print(f"  Scenes:   {result.segments_detected}")
        console.print(f"  Clips:    {result.clips_rendered}")
    except ShowtimeError as e:
        console.print(f"\n[red]Pipeline error:[/] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
