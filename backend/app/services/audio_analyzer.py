"""Audio transcription with sentence-level segmentation.

Supports two providers:
- Groq Whisper (cloud, default) — fast (~10x realtime), free tier: 28,800s audio/day
- Local Whisper (offline) — no API key needed, but slower on CPU

Set via SHOWTIME_WHISPER_PROVIDER=groq|local and the corresponding settings.
"""

from __future__ import annotations

import re
from pathlib import Path

import whisper
from groq import Groq

from app.core.config import Settings
from app.core.exceptions import TranscriptionError
from app.models.domain import VoiceoverSentence

# Sentence-ending punctuation pattern
_SENTENCE_END = re.compile(r"[.!?]$")


def transcribe_audio(
    audio_path: Path,
    settings: Settings | None = None,
) -> list[VoiceoverSentence]:
    """Transcribe audio and return sentence-level segments with timestamps.

    Routes to Groq Whisper API (default) or local Whisper based on settings.

    Args:
        audio_path: Path to the voiceover audio file.
        settings: Optional settings override; uses defaults if None.

    Returns:
        List of VoiceoverSentence objects ordered by start time.

    Raises:
        TranscriptionError: If the audio cannot be transcribed.
    """
    if settings is None:
        settings = Settings()

    if settings.whisper_provider == "groq":
        words = _transcribe_groq(audio_path, settings)
    else:
        words = _transcribe_local(audio_path, settings)

    if not words:
        return []

    return _group_into_sentences(words)


def _transcribe_groq(audio_path: Path, settings: Settings) -> list[dict]:
    """Transcribe using Groq's Whisper API.

    Returns word-level dicts with 'word', 'start', 'end' keys.
    """
    if not settings.groq_api_key:
        raise TranscriptionError(
            "Groq API key not set. Set SHOWTIME_GROQ_API_KEY or switch to "
            "SHOWTIME_WHISPER_PROVIDER=local for offline transcription."
        )

    try:
        client = Groq(api_key=settings.groq_api_key)

        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(audio_path.name, audio_file),
                model=settings.groq_whisper_model,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )

        return _extract_words_groq(response)
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"Groq transcription failed for {audio_path}: {e}") from e


def _extract_words_groq(response) -> list[dict]:
    """Extract word-level entries from Groq Whisper response.

    Groq may return words as objects (with .word attribute) or dicts
    (with ["word"] key) depending on the SDK version. Handle both.
    """
    words: list[dict] = []

    def _get(obj, key):
        """Get a value from either an object attribute or dict key."""
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _extract_word(w) -> dict | None:
        word = _get(w, "word")
        start = _get(w, "start")
        end = _get(w, "end")
        if word is not None and start is not None and end is not None:
            return {"word": str(word).strip(), "start": float(start), "end": float(end)}
        return None

    # Try response.words first
    raw_words = _get(response, "words")
    if raw_words:
        for w in raw_words:
            parsed = _extract_word(w)
            if parsed:
                words.append(parsed)

    # Fallback: try segments → words
    if not words:
        raw_segments = _get(response, "segments")
        if raw_segments:
            for seg in raw_segments:
                seg_words = _get(seg, "words")
                if seg_words:
                    for w in seg_words:
                        parsed = _extract_word(w)
                        if parsed:
                            words.append(parsed)

    return words


def _transcribe_local(audio_path: Path, settings: Settings) -> list[dict]:
    """Transcribe using local Whisper model.

    Returns word-level dicts with 'word', 'start', 'end' keys.
    """
    try:
        model = whisper.load_model(settings.whisper_model)
        result = model.transcribe(str(audio_path), word_timestamps=True)
    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe {audio_path}: {e}") from e

    return _extract_words_local(result)


def _extract_words_local(result: dict) -> list[dict]:
    """Flatten all word-level entries from local Whisper's output."""
    words: list[dict] = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"],
            })
    return words


def _group_into_sentences(words: list[dict]) -> list[VoiceoverSentence]:
    """Group word-level entries into sentence-level segments.

    Splits on sentence-ending punctuation (. ! ?).
    If the transcript has no punctuation, all words form a single sentence.
    """
    sentences: list[VoiceoverSentence] = []
    current_words: list[dict] = []
    sentence_id = 1

    for word in words:
        current_words.append(word)

        if _SENTENCE_END.search(word["word"]):
            sentence = _build_sentence(sentence_id, current_words)
            sentences.append(sentence)
            sentence_id += 1
            current_words = []

    # Remaining words that didn't end with punctuation
    if current_words:
        sentence = _build_sentence(sentence_id, current_words)
        sentences.append(sentence)

    return sentences


def _build_sentence(sentence_id: int, words: list[dict]) -> VoiceoverSentence:
    """Create a VoiceoverSentence from a list of word dicts."""
    text = " ".join(w["word"] for w in words)
    return VoiceoverSentence(
        sentence_id=sentence_id,
        text=text,
        start=words[0]["start"],
        end=words[-1]["end"],
    )
