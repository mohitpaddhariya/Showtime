"""Tests for the AI mapper pipeline component."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import MappingError
from app.models.domain import CaptionedSegment, MappingEntry, VoiceoverSentence
from app.services.ai_mapper import (
    _fallback_time_mapping,
    _find_best_overlap,
    _parse_response,
    map_sentences_to_segments,
)


# ── Shared fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_segments() -> list[CaptionedSegment]:
    return [
        CaptionedSegment(segment_id=1, start=0.0, end=5.0, description="Landing page"),
        CaptionedSegment(segment_id=2, start=5.0, end=10.0, description="Sign up form"),
        CaptionedSegment(segment_id=3, start=10.0, end=15.0, description="Dashboard"),
    ]


@pytest.fixture
def sample_sentences() -> list[VoiceoverSentence]:
    return [
        VoiceoverSentence(sentence_id=1, text="Here is our landing page.", start=0.0, end=3.0),
        VoiceoverSentence(sentence_id=2, text="Users can sign up here.", start=3.0, end=6.0),
    ]


VALID_LLM_RESPONSE = (
    '[{"sentence_id": 1, "segment_id": 1, "speed_factor": 1.0},'
    ' {"sentence_id": 2, "segment_id": 2, "speed_factor": 1.5}]'
)

VALID_LLM_RESPONSE_WRAPPED = (
    '{"mappings": [{"sentence_id": 1, "segment_id": 1, "speed_factor": 1.0},'
    ' {"sentence_id": 2, "segment_id": 2, "speed_factor": 1.5}]}'
)


def _mock_groq_response(content: str) -> MagicMock:
    """Build a mock Groq chat completion response."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ── Tests for Groq provider (default) ────────────────────────────────


class TestGroqProvider:
    def _mock_groq(self, mocker, content: str):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(content)
        mocker.patch("app.services.ai_mapper.Groq", return_value=mock_client)
        return mock_client

    def test_valid_mapping_via_groq(self, mocker, sample_segments, sample_sentences):
        self._mock_groq(mocker, VALID_LLM_RESPONSE)
        settings = Settings(llm_provider="groq", groq_api_key="test-key")

        result = map_sentences_to_segments(sample_segments, sample_sentences, settings)
        assert len(result) == 2
        assert result[0].sentence_id == 1
        assert result[1].speed_factor == 1.5

    def test_groq_wrapped_response(self, mocker, sample_segments, sample_sentences):
        self._mock_groq(mocker, VALID_LLM_RESPONSE_WRAPPED)
        settings = Settings(llm_provider="groq", groq_api_key="test-key")

        result = map_sentences_to_segments(sample_segments, sample_sentences, settings)
        assert len(result) == 2

    def test_groq_uses_correct_model(self, mocker, sample_segments, sample_sentences):
        client = self._mock_groq(mocker, VALID_LLM_RESPONSE)
        settings = Settings(
            llm_provider="groq", groq_api_key="test-key", groq_model="llama-3.1-8b-instant"
        )

        map_sentences_to_segments(sample_segments, sample_sentences, settings)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.1-8b-instant"

    def test_groq_missing_api_key_falls_back(self, mocker, sample_segments, sample_sentences):
        """Without API key, Groq raises MappingError, which triggers fallback."""
        settings = Settings(llm_provider="groq", groq_api_key="")
        result = map_sentences_to_segments(sample_segments, sample_sentences, settings)
        # Falls back to time-based mapping
        assert len(result) == len(sample_sentences)

    def test_groq_connection_failure_falls_back(self, mocker, sample_segments, sample_sentences):
        mocker.patch(
            "app.services.ai_mapper.Groq",
            side_effect=ConnectionError("Network error"),
        )
        settings = Settings(llm_provider="groq", groq_api_key="test-key")
        result = map_sentences_to_segments(sample_segments, sample_sentences, settings)
        assert len(result) == len(sample_sentences)

    def test_groq_uses_low_temperature(self, mocker, sample_segments, sample_sentences):
        client = self._mock_groq(mocker, VALID_LLM_RESPONSE)
        settings = Settings(llm_provider="groq", groq_api_key="test-key")

        map_sentences_to_segments(sample_segments, sample_sentences, settings)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1

    def test_groq_requests_json_format(self, mocker, sample_segments, sample_sentences):
        client = self._mock_groq(mocker, VALID_LLM_RESPONSE)
        settings = Settings(llm_provider="groq", groq_api_key="test-key")

        map_sentences_to_segments(sample_segments, sample_sentences, settings)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}


# ── Tests for Ollama provider ─────────────────────────────────────────


class TestOllamaProvider:
    def test_valid_mapping_via_ollama(self, mocker, sample_segments, sample_sentences):
        mock_response = {"message": {"content": VALID_LLM_RESPONSE}}
        mocker.patch("app.services.ai_mapper.ollama.chat", return_value=mock_response)
        settings = Settings(llm_provider="ollama")

        result = map_sentences_to_segments(sample_segments, sample_sentences, settings)
        assert len(result) == 2
        assert result[0].sentence_id == 1

    def test_ollama_uses_correct_model(self, mocker, sample_segments, sample_sentences):
        mock_response = {"message": {"content": VALID_LLM_RESPONSE}}
        chat_mock = mocker.patch(
            "app.services.ai_mapper.ollama.chat", return_value=mock_response
        )
        settings = Settings(llm_provider="ollama", ollama_model="llama3.1")

        map_sentences_to_segments(sample_segments, sample_sentences, settings)
        assert chat_mock.call_args.kwargs["model"] == "llama3.1"

    def test_ollama_connection_failure_falls_back(self, mocker, sample_segments, sample_sentences):
        mocker.patch(
            "app.services.ai_mapper.ollama.chat",
            side_effect=ConnectionError("Connection refused"),
        )
        settings = Settings(llm_provider="ollama")
        result = map_sentences_to_segments(sample_segments, sample_sentences, settings)
        assert len(result) == len(sample_sentences)


# ── Tests for shared behavior (provider-agnostic) ─────────────────────


class TestMapSentencesToSegments:
    def test_fallback_on_invalid_json(self, mocker, sample_segments, sample_sentences):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response("not valid json")
        mocker.patch("app.services.ai_mapper.Groq", return_value=mock_client)
        settings = Settings(llm_provider="groq", groq_api_key="test-key")

        result = map_sentences_to_segments(sample_segments, sample_sentences, settings)
        assert len(result) == len(sample_sentences)

    def test_fallback_on_incomplete_mapping(self, mocker, sample_segments, sample_sentences):
        incomplete = '[{"sentence_id": 1, "segment_id": 1, "speed_factor": 1.0}]'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(incomplete)
        mocker.patch("app.services.ai_mapper.Groq", return_value=mock_client)
        settings = Settings(llm_provider="groq", groq_api_key="test-key")

        result = map_sentences_to_segments(sample_segments, sample_sentences, settings)
        assert len(result) == len(sample_sentences)

    def test_empty_sentences(self, sample_segments):
        result = map_sentences_to_segments(sample_segments, [])
        assert result == []

    def test_no_segments_raises(self, sample_sentences):
        with pytest.raises(MappingError, match="No screen segments"):
            map_sentences_to_segments([], sample_sentences)

    def test_results_sorted_by_sentence_id(self, mocker, sample_segments, sample_sentences):
        reversed_response = (
            '[{"sentence_id": 2, "segment_id": 2, "speed_factor": 1.5},'
            ' {"sentence_id": 1, "segment_id": 1, "speed_factor": 1.0}]'
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(reversed_response)
        mocker.patch("app.services.ai_mapper.Groq", return_value=mock_client)
        settings = Settings(llm_provider="groq", groq_api_key="test-key")

        result = map_sentences_to_segments(sample_segments, sample_sentences, settings)
        assert result[0].sentence_id == 1
        assert result[1].sentence_id == 2


# ── Tests for _parse_response ─────────────────────────────────────────


class TestParseResponse:
    def test_valid_array(self, sample_segments, sample_sentences):
        entries = _parse_response(VALID_LLM_RESPONSE, sample_segments, sample_sentences)
        assert len(entries) == 2

    def test_wrapped_object(self, sample_segments, sample_sentences):
        entries = _parse_response(VALID_LLM_RESPONSE_WRAPPED, sample_segments, sample_sentences)
        assert len(entries) == 2

    def test_invalid_json_raises(self, sample_segments, sample_sentences):
        with pytest.raises(Exception):
            _parse_response("not json", sample_segments, sample_sentences)

    def test_invalid_segment_id_raises(self, sample_segments, sample_sentences):
        bad = '[{"sentence_id": 1, "segment_id": 999, "speed_factor": 1.0}, {"sentence_id": 2, "segment_id": 1, "speed_factor": 1.0}]'
        with pytest.raises(MappingError, match="Invalid segment_id"):
            _parse_response(bad, sample_segments, sample_sentences)

    def test_missing_sentence_raises(self, sample_segments, sample_sentences):
        incomplete = '[{"sentence_id": 1, "segment_id": 1, "speed_factor": 1.0}]'
        with pytest.raises(MappingError, match="Missing sentence_ids"):
            _parse_response(incomplete, sample_segments, sample_sentences)

    def test_default_speed_factor(self, sample_segments, sample_sentences):
        no_speed = '[{"sentence_id": 1, "segment_id": 1}, {"sentence_id": 2, "segment_id": 2}]'
        entries = _parse_response(no_speed, sample_segments, sample_sentences)
        assert all(e.speed_factor == 1.0 for e in entries)

    def test_unexpected_dict_structure_raises(self, sample_segments, sample_sentences):
        bad_dict = '{"foo": "bar"}'
        with pytest.raises(MappingError, match="Unexpected JSON structure"):
            _parse_response(bad_dict, sample_segments, sample_sentences)


# ── Tests for _fallback_time_mapping ──────────────────────────────────


class TestChronologicalMapping:
    """Tests for the chronological (fallback) mapping strategy."""

    def test_maps_all_sentences(self, sample_segments, sample_sentences):
        result = _fallback_time_mapping(sample_segments, sample_sentences)
        assert len(result) == len(sample_sentences)

    def test_single_sentence_maps_to_middle_segment(self, sample_segments):
        """One sentence → maps to segment at scaled midpoint of video."""
        sentences = [VoiceoverSentence(sentence_id=1, text="test", start=0.0, end=5.0)]
        result = _fallback_time_mapping(sample_segments, sentences)
        # scale = 15/5 = 3.0. Midpoint audio=2.5 → video_time = 0 + 2.5*3 = 7.5 → seg 2 (5-10)
        assert result[0].segment_id == 2

    def test_speed_factor_always_one(self, sample_segments, sample_sentences):
        """Chronological mapper sets speed_factor=1.0 (renderer calculates real speed)."""
        result = _fallback_time_mapping(sample_segments, sample_sentences)
        assert all(e.speed_factor == 1.0 for e in result)

    def test_multi_sentence_distributes_across_segments(self, sample_segments):
        """Multiple sentences should map to different segments chronologically."""
        sentences = [
            VoiceoverSentence(sentence_id=1, text="first", start=0.0, end=5.0),
            VoiceoverSentence(sentence_id=2, text="second", start=5.0, end=10.0),
            VoiceoverSentence(sentence_id=3, text="third", start=10.0, end=15.0),
        ]
        result = _fallback_time_mapping(sample_segments, sentences)
        # Equal-duration sentences should spread across all 3 segments
        seg_ids = [e.segment_id for e in result]
        assert seg_ids == [1, 2, 3]

    def test_preserves_sentence_order(self, sample_segments, sample_sentences):
        result = _fallback_time_mapping(sample_segments, sample_sentences)
        ids = [e.sentence_id for e in result]
        assert ids == sorted(ids)


# ── Tests for _find_best_overlap ──────────────────────────────────────


class TestFindBestOverlap:
    def test_exact_overlap(self, sample_segments):
        sentence = VoiceoverSentence(sentence_id=1, text="t", start=0.0, end=5.0)
        best = _find_best_overlap(sentence, sample_segments)
        assert best.segment_id == 1

    def test_partial_overlap(self, sample_segments):
        sentence = VoiceoverSentence(sentence_id=1, text="t", start=4.0, end=8.0)
        best = _find_best_overlap(sentence, sample_segments)
        assert best.segment_id == 2

    def test_no_overlap_nearest(self, sample_segments):
        sentence = VoiceoverSentence(sentence_id=1, text="t", start=100.0, end=105.0)
        best = _find_best_overlap(sentence, sample_segments)
        assert best.segment_id == 3
