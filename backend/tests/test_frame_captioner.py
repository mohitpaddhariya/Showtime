"""Tests for the frame captioner pipeline component."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.exceptions import CaptionError
from app.models.domain import CaptionedSegment, VideoSegment
from app.services.frame_captioner import _ocr_keyframe, _describe_keyframe, caption_segments


@pytest.fixture
def dummy_keyframe(tmp_path: Path) -> Path:
    """Create a minimal PNG file so path.exists() returns True."""
    kf = tmp_path / "dummy.png"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(kf), img)
    return kf


# ── Unit tests (mock OCR + structural analysis) ──────────────────────


class TestCaptionSegmentsUnit:
    def _mock_describe(self, mocker, description: str):
        """Mock the entire _describe_keyframe function."""
        mocker.patch("app.services.frame_captioner._describe_keyframe", return_value=description)

    def test_returns_correct_count(self, mocker, dummy_keyframe):
        self._mock_describe(mocker, "Login Page")
        segments = [
            VideoSegment(segment_id=1, start=0, end=5, keyframe_path=dummy_keyframe),
            VideoSegment(segment_id=2, start=5, end=10, keyframe_path=dummy_keyframe),
        ]
        result = caption_segments(segments)
        assert len(result) == 2
        assert all(isinstance(r, CaptionedSegment) for r in result)

    def test_description_set(self, mocker, dummy_keyframe):
        self._mock_describe(mocker, "Text on screen: Dashboard | Visual: light background")
        segments = [VideoSegment(segment_id=1, start=0, end=5, keyframe_path=dummy_keyframe)]
        result = caption_segments(segments)
        assert "Dashboard" in result[0].description

    def test_preserves_segment_fields(self, mocker, dummy_keyframe):
        self._mock_describe(mocker, "text")
        segments = [VideoSegment(segment_id=7, start=2.5, end=8.3, is_idle=True, keyframe_path=dummy_keyframe)]
        result = caption_segments(segments)
        assert result[0].segment_id == 7
        assert result[0].start == 2.5
        assert result[0].end == 8.3
        assert result[0].is_idle is True
        assert result[0].keyframe_path == dummy_keyframe

    def test_empty_description_for_blank_image(self, mocker, dummy_keyframe):
        self._mock_describe(mocker, "")
        segments = [VideoSegment(segment_id=1, start=0, end=5, keyframe_path=dummy_keyframe)]
        result = caption_segments(segments)
        assert result[0].description == ""


class TestNoneKeyframe:
    def test_none_keyframe_gives_empty_description(self):
        segments = [VideoSegment(segment_id=1, start=0, end=5, keyframe_path=None)]
        result = caption_segments(segments)
        assert result[0].description == ""

    def test_nonexistent_keyframe_gives_empty_description(self):
        segments = [
            VideoSegment(segment_id=1, start=0, end=5, keyframe_path=Path("/does/not/exist.png"))
        ]
        result = caption_segments(segments)
        assert result[0].description == ""

    def test_mixed_keyframes(self, mocker, dummy_keyframe):
        def _mock_describe(path):
            if path is None or not path.exists():
                return ""
            return "Found text"

        mocker.patch("app.services.frame_captioner._describe_keyframe", side_effect=_mock_describe)
        segments = [
            VideoSegment(segment_id=1, start=0, end=5, keyframe_path=None),
            VideoSegment(segment_id=2, start=5, end=10, keyframe_path=dummy_keyframe),
        ]
        result = caption_segments(segments)
        assert result[0].description == ""
        assert result[1].description == "Found text"


# ── Tests for OCR function ────────────────────────────────────────────


class TestOcrKeyframe:
    def test_extracts_text(self, mocker, dummy_keyframe):
        mocker.patch("app.services.frame_captioner.pytesseract.image_to_string", return_value="Hello World")
        result = _ocr_keyframe(dummy_keyframe)
        assert result == "Hello World"

    def test_whitespace_cleaned(self, mocker, dummy_keyframe):
        mocker.patch(
            "app.services.frame_captioner.pytesseract.image_to_string",
            return_value="  Hello  \n\n  World  \n\n",
        )
        result = _ocr_keyframe(dummy_keyframe)
        assert result == "Hello\nWorld"


# ── Tests for describe_keyframe (OCR + structural) ────────────────────


class TestDescribeKeyframe:
    def test_combines_ocr_and_visual(self, mocker, dummy_keyframe):
        mocker.patch("app.services.frame_captioner.pytesseract.image_to_string", return_value="Login")
        result = _describe_keyframe(dummy_keyframe)
        assert "Text on screen: Login" in result
        assert "Visual:" in result

    def test_visual_only_when_no_text(self, mocker, dummy_keyframe):
        mocker.patch("app.services.frame_captioner.pytesseract.image_to_string", return_value="   ")
        result = _describe_keyframe(dummy_keyframe)
        # No OCR text, but visual analysis should still work
        assert "Visual:" in result or result == ""

    def test_error_raises_caption_error(self, mocker, dummy_keyframe):
        mocker.patch(
            "app.services.frame_captioner.Image.open",
            side_effect=Exception("corrupt"),
        )
        with pytest.raises(CaptionError, match="Captioning failed"):
            _describe_keyframe(dummy_keyframe)

    def test_none_path_returns_empty(self):
        assert _describe_keyframe(None) == ""


# ── Integration test (requires Tesseract binary) ─────────────────────


@pytest.mark.integration
class TestCaptionIntegration:
    def test_full_caption_on_text_image(self, tmp_path: Path):
        """Create an image with text and verify captioning picks it up."""
        img = np.full((200, 400, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Hello World", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        kf_path = tmp_path / "text_frame.png"
        cv2.imwrite(str(kf_path), img)

        segments = [VideoSegment(segment_id=1, start=0, end=5, keyframe_path=kf_path)]
        result = caption_segments(segments)
        assert len(result[0].description) > 0
        # Should contain OCR text and visual info
        desc = result[0].description.lower()
        assert "hello" in desc or "world" in desc
