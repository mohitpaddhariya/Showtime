"""Centralised configuration with sensible defaults."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Scene detection
    scene_threshold: float = 30.0  # pixel-diff threshold for scene change
    scene_refine_threshold: float = 20.0  # lower threshold for splitting long segments
    max_segment_duration: float = 8.0  # auto-split segments longer than this (seconds)
    min_segment_duration: float = 1.5  # merge segments shorter than this into neighbors
    idle_threshold: float = 2.0  # pixel-diff threshold for "idle"
    idle_min_duration: float = 1.5  # seconds of low-delta to count as idle
    sample_fps: int = 4  # frames per second to sample (higher = catches faster transitions)

    # Whisper provider: "groq" (cloud, default) or "local" (offline)
    whisper_provider: str = "groq"
    whisper_model: str = "base"  # local Whisper: tiny | base | small | medium | large
    groq_whisper_model: str = "whisper-large-v3-turbo"  # Groq Whisper model

    # LLM provider: "groq" (cloud, default) or "ollama" (local)
    llm_provider: str = "groq"

    # Groq (free tier: 30 req/min, 14400 req/day)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"  # text-only fallback
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # vision model (sees keyframes)

    # Ollama (local fallback)
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"

    # Renderer
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    output_preset: str = "medium"  # FFmpeg encoding preset
    crf: int = 23  # Constant Rate Factor
    crossfade_duration: float = 0.3  # seconds of crossfade between clips (0 = hard cut)
    max_playback_speed: float = 2.5  # max speed before auto-freeze (>2.5x looks choppy)
    min_playback_speed: float = 0.5  # min speed before auto-freeze (<0.5x looks frozen)

    model_config = {"env_prefix": "SHOWTIME_", "env_file": ".env", "env_file_encoding": "utf-8"}
