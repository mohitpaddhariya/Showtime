"""Tests for the audio analyzer pipeline component."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import TranscriptionError
from app.models.domain import VoiceoverSentence
from app.services.audio_analyzer import (
    _build_sentence,
    _extract_words_local,
    _extract_words_groq,
    _group_into_sentences,
    transcribe_audio,
)


# ── Mock data ─────────────────────────────────────────────────────────

MOCK_LOCAL_RESULT_TWO_SENTENCES = {
    "segments": [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "Here is our landing page. Users can sign up easily.",
            "words": [
                {"word": "Here", "start": 0.0, "end": 0.3},
                {"word": "is", "start": 0.3, "end": 0.5},
                {"word": "our", "start": 0.5, "end": 0.7},
                {"word": "landing", "start": 0.7, "end": 1.0},
                {"word": "page.", "start": 1.0, "end": 1.3},
                {"word": "Users", "start": 1.4, "end": 1.6},
                {"word": "can", "start": 1.6, "end": 1.7},
                {"word": "sign", "start": 1.7, "end": 1.9},
                {"word": "up", "start": 1.9, "end": 2.0},
                {"word": "easily.", "start": 2.0, "end": 2.5},
            ],
        }
    ]
}

MOCK_LOCAL_QUESTION = {
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "What is this? It's a demo.",
            "words": [
                {"word": "What", "start": 0.0, "end": 0.2},
                {"word": "is", "start": 0.2, "end": 0.4},
                {"word": "this?", "start": 0.4, "end": 0.7},
                {"word": "It's", "start": 0.8, "end": 1.0},
                {"word": "a", "start": 1.0, "end": 1.1},
                {"word": "demo.", "start": 1.1, "end": 1.5},
            ],
        }
    ]
}

MOCK_LOCAL_NO_PUNCTUATION = {
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world this is a test",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.3},
                {"word": "world", "start": 0.3, "end": 0.6},
                {"word": "this", "start": 0.7, "end": 0.9},
                {"word": "is", "start": 0.9, "end": 1.0},
                {"word": "a", "start": 1.0, "end": 1.1},
                {"word": "test", "start": 1.1, "end": 1.5},
            ],
        }
    ]
}

MOCK_LOCAL_EMPTY = {"segments": []}

MOCK_LOCAL_NO_WORDS = {
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}]
}


def _make_groq_word(word: str, start: float, end: float) -> MagicMock:
    m = MagicMock()
    m.word = word
    m.start = start
    m.end = end
    return m


def _make_groq_response(words: list[tuple[str, float, float]]) -> MagicMock:
    """Build a mock Groq Whisper response with word-level data."""
    mock = MagicMock()
    mock.words = [_make_groq_word(w, s, e) for w, s, e in words]
    mock.segments = None
    return mock


# ── Tests for local Whisper provider ──────────────────────────────────


class TestLocalWhisperProvider:
    def _mock_local(self, mocker, result: dict):
        mock_model = mocker.MagicMock()
        mock_model.transcribe.return_value = result
        mocker.patch("app.services.audio_analyzer.whisper.load_model", return_value=mock_model)
        return mock_model

    def test_two_sentences(self, mocker):
        self._mock_local(mocker, MOCK_LOCAL_RESULT_TWO_SENTENCES)
        settings = Settings(whisper_provider="local")
        sentences = transcribe_audio(Path("fake.wav"), settings)
        assert len(sentences) == 2

    def test_sentence_text(self, mocker):
        self._mock_local(mocker, MOCK_LOCAL_RESULT_TWO_SENTENCES)
        settings = Settings(whisper_provider="local")
        sentences = transcribe_audio(Path("fake.wav"), settings)
        assert sentences[0].text == "Here is our landing page."
        assert sentences[1].text == "Users can sign up easily."

    def test_timestamps(self, mocker):
        self._mock_local(mocker, MOCK_LOCAL_RESULT_TWO_SENTENCES)
        settings = Settings(whisper_provider="local")
        sentences = transcribe_audio(Path("fake.wav"), settings)
        assert sentences[0].start == pytest.approx(0.0)
        assert sentences[0].end == pytest.approx(1.3)

    def test_empty_result(self, mocker):
        self._mock_local(mocker, MOCK_LOCAL_EMPTY)
        settings = Settings(whisper_provider="local")
        assert transcribe_audio(Path("fake.wav"), settings) == []

    def test_no_words(self, mocker):
        self._mock_local(mocker, MOCK_LOCAL_NO_WORDS)
        settings = Settings(whisper_provider="local")
        assert transcribe_audio(Path("fake.wav"), settings) == []

    def test_failure_raises(self, mocker):
        mocker.patch(
            "app.services.audio_analyzer.whisper.load_model",
            side_effect=RuntimeError("model not found"),
        )
        settings = Settings(whisper_provider="local")
        with pytest.raises(TranscriptionError, match="Failed to transcribe"):
            transcribe_audio(Path("fake.wav"), settings)

    def test_uses_model_setting(self, mocker):
        mock_model = self._mock_local(mocker, MOCK_LOCAL_EMPTY)
        load_mock = mocker.patch(
            "app.services.audio_analyzer.whisper.load_model", return_value=mock_model
        )
        settings = Settings(whisper_provider="local", whisper_model="tiny")
        transcribe_audio(Path("fake.wav"), settings)
        load_mock.assert_called_once_with("tiny")


# ── Tests for Groq Whisper provider ───────────────────────────────────


class TestGroqWhisperProvider:
    def _mock_groq(self, mocker, words: list[tuple[str, float, float]]):
        response = _make_groq_response(words)
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = response
        mocker.patch("app.services.audio_analyzer.Groq", return_value=mock_client)
        return mock_client

    def test_two_sentences(self, mocker):
        self._mock_groq(mocker, [
            ("Here", 0.0, 0.3), ("is", 0.3, 0.5), ("our", 0.5, 0.7),
            ("landing", 0.7, 1.0), ("page.", 1.0, 1.3),
            ("Users", 1.4, 1.6), ("can", 1.6, 1.7), ("sign", 1.7, 1.9),
            ("up", 1.9, 2.0), ("easily.", 2.0, 2.5),
        ])
        settings = Settings(whisper_provider="groq", groq_api_key="test-key")

        # Need a real file for open()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sentences = transcribe_audio(Path(f.name), settings)

        assert len(sentences) == 2
        assert sentences[0].text == "Here is our landing page."
        assert sentences[1].text == "Users can sign up easily."

    def test_timestamps(self, mocker):
        self._mock_groq(mocker, [
            ("Hello.", 0.0, 0.5), ("World.", 0.6, 1.0),
        ])
        settings = Settings(whisper_provider="groq", groq_api_key="test-key")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sentences = transcribe_audio(Path(f.name), settings)

        assert sentences[0].start == 0.0
        assert sentences[0].end == 0.5
        assert sentences[1].start == 0.6

    def test_missing_api_key_raises(self):
        settings = Settings(whisper_provider="groq", groq_api_key="")
        with pytest.raises(TranscriptionError, match="Groq API key not set"):
            transcribe_audio(Path("fake.wav"), settings)

    def test_connection_failure_raises(self, mocker):
        mocker.patch(
            "app.services.audio_analyzer.Groq",
            side_effect=ConnectionError("Network error"),
        )
        settings = Settings(whisper_provider="groq", groq_api_key="test-key")
        with pytest.raises(TranscriptionError, match="Groq transcription failed"):
            transcribe_audio(Path("fake.wav"), settings)

    def test_uses_correct_model(self, mocker):
        client = self._mock_groq(mocker, [("Hi.", 0.0, 0.5)])
        settings = Settings(
            whisper_provider="groq", groq_api_key="test-key",
            groq_whisper_model="whisper-large-v3",
        )

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            transcribe_audio(Path(f.name), settings)

        call_kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-large-v3"


# ── Tests for _extract_words_local ────────────────────────────────────


class TestExtractWordsLocal:
    def test_extracts_all_words(self):
        words = _extract_words_local(MOCK_LOCAL_RESULT_TWO_SENTENCES)
        assert len(words) == 10

    def test_word_structure(self):
        words = _extract_words_local(MOCK_LOCAL_RESULT_TWO_SENTENCES)
        assert words[0] == {"word": "Here", "start": 0.0, "end": 0.3}

    def test_strips_whitespace(self):
        result = {"segments": [{"words": [{"word": " hello ", "start": 0.0, "end": 0.5}]}]}
        words = _extract_words_local(result)
        assert words[0]["word"] == "hello"

    def test_empty_segments(self):
        assert _extract_words_local({"segments": []}) == []

    def test_missing_segments_key(self):
        assert _extract_words_local({}) == []


# ── Tests for _extract_words_groq ─────────────────────────────────────


class TestExtractWordsGroq:
    def test_extracts_from_words_attr(self):
        response = _make_groq_response([("Hello", 0.0, 0.3), ("world.", 0.3, 0.6)])
        words = _extract_words_groq(response)
        assert len(words) == 2
        assert words[0] == {"word": "Hello", "start": 0.0, "end": 0.3}

    def test_empty_words(self):
        response = MagicMock()
        response.words = []
        response.segments = None
        words = _extract_words_groq(response)
        assert words == []

    def test_no_words_attr(self):
        response = MagicMock()
        response.words = None
        response.segments = None
        words = _extract_words_groq(response)
        assert words == []


# ── Tests for _group_into_sentences (shared logic) ────────────────────


class TestGroupIntoSentences:
    def test_period_split(self):
        words = _extract_words_local(MOCK_LOCAL_RESULT_TWO_SENTENCES)
        sentences = _group_into_sentences(words)
        assert len(sentences) == 2

    def test_question_mark_split(self):
        words = _extract_words_local(MOCK_LOCAL_QUESTION)
        sentences = _group_into_sentences(words)
        assert len(sentences) == 2
        assert sentences[0].text == "What is this?"
        assert sentences[1].text == "It's a demo."

    def test_no_punctuation_single_sentence(self):
        words = _extract_words_local(MOCK_LOCAL_NO_PUNCTUATION)
        sentences = _group_into_sentences(words)
        assert len(sentences) == 1

    def test_timestamps_monotonic(self):
        words = _extract_words_local(MOCK_LOCAL_RESULT_TWO_SENTENCES)
        sentences = _group_into_sentences(words)
        for i in range(1, len(sentences)):
            assert sentences[i].start >= sentences[i - 1].end

    def test_empty_words(self):
        assert _group_into_sentences([]) == []


# ── Tests for _build_sentence ─────────────────────────────────────────


class TestBuildSentence:
    def test_joins_words(self):
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.3},
            {"word": "world.", "start": 0.3, "end": 0.6},
        ]
        sentence = _build_sentence(1, words)
        assert sentence.text == "Hello world."
        assert sentence.sentence_id == 1
        assert sentence.start == 0.0
        assert sentence.end == 0.6

    def test_single_word(self):
        words = [{"word": "Hi.", "start": 0.0, "end": 0.5}]
        sentence = _build_sentence(1, words)
        assert sentence.text == "Hi."
        assert sentence.duration == 0.5
