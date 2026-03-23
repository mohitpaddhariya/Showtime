"""Live integration test for Groq API — run manually to verify LLM mapping works.

Usage:
    uv run pytest tests/test_groq_live.py -v -s

Requires SHOWTIME_GROQ_API_KEY in .env or environment.
"""

import json

import pytest

from app.core.config import Settings
from app.models.domain import CaptionedSegment, VoiceoverSentence
from app.services.ai_mapper import map_sentences_to_segments, _call_groq


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    if not s.groq_api_key:
        pytest.skip("SHOWTIME_GROQ_API_KEY not set — skipping live test")
    return s


@pytest.fixture
def demo_segments() -> list[CaptionedSegment]:
    """Simulate a 3-scene screen recording of a web app demo."""
    return [
        CaptionedSegment(
            segment_id=1, start=0.0, end=8.0,
            description="Landing page with hero text: 'Welcome to TaskFlow' and a 'Get Started' button",
        ),
        CaptionedSegment(
            segment_id=2, start=8.0, end=18.0,
            description="Sign up form with email field, password field, and 'Create Account' button",
        ),
        CaptionedSegment(
            segment_id=3, start=18.0, end=30.0,
            description="Dashboard showing task list with 3 tasks, a sidebar, and 'Add Task' button",
        ),
    ]


@pytest.fixture
def demo_sentences() -> list[VoiceoverSentence]:
    """Simulate voiceover narration for the demo."""
    return [
        VoiceoverSentence(
            sentence_id=1,
            text="Welcome to TaskFlow, the simplest way to manage your daily tasks.",
            start=0.0, end=4.5,
        ),
        VoiceoverSentence(
            sentence_id=2,
            text="Getting started is easy, just enter your email and create an account.",
            start=4.5, end=9.0,
        ),
        VoiceoverSentence(
            sentence_id=3,
            text="Once you're in, you'll see your dashboard with all your tasks organized.",
            start=9.0, end=14.0,
        ),
    ]


class TestGroqLive:
    """Live tests that hit the real Groq API."""

    def test_raw_api_call(self, settings, demo_segments, demo_sentences):
        """Verify we can reach Groq and get a JSON response."""
        raw = _call_groq(demo_segments, demo_sentences, settings)

        print(f"\n--- Raw Groq response ---\n{raw}\n---")

        data = json.loads(raw)
        assert data is not None

    def test_full_mapping(self, settings, demo_segments, demo_sentences):
        """End-to-end: segments + sentences → validated MappingEntry list."""
        result = map_sentences_to_segments(demo_segments, demo_sentences, settings)

        print("\n--- Mapping results ---")
        for entry in result:
            seg = next(s for s in demo_segments if s.segment_id == entry.segment_id)
            sent = next(s for s in demo_sentences if s.sentence_id == entry.sentence_id)
            print(
                f"  Sentence {entry.sentence_id}: \"{sent.text[:50]}...\"\n"
                f"    → Segment {entry.segment_id}: \"{seg.description[:50]}...\"\n"
                f"    → Speed: {entry.speed_factor}x"
            )

        # Every sentence must be mapped
        assert len(result) == len(demo_sentences)
        # All sentence IDs covered
        assert {e.sentence_id for e in result} == {1, 2, 3}
        # All segment IDs are valid
        valid_ids = {s.segment_id for s in demo_segments}
        assert all(e.segment_id in valid_ids for e in result)
        # Speed factors are reasonable
        assert all(0.25 <= e.speed_factor <= 4.0 for e in result)

    def test_smart_matching(self, settings, demo_segments, demo_sentences):
        """Verify the LLM actually maps semantically, not just by time."""
        result = map_sentences_to_segments(demo_segments, demo_sentences, settings)

        mapping = {e.sentence_id: e.segment_id for e in result}
        print(f"\n--- Mapping: {mapping}")

        # "Welcome to TaskFlow" should map to the landing page (segment 1)
        assert mapping[1] == 1, "Expected 'Welcome' sentence to map to landing page"
        # "enter your email" should map to sign up form (segment 2)
        assert mapping[2] == 2, "Expected 'sign up' sentence to map to sign up form"
        # "dashboard with all your tasks" should map to dashboard (segment 3)
        assert mapping[3] == 3, "Expected 'dashboard' sentence to map to dashboard"

    def test_larger_input(self, settings):
        """Test with more segments/sentences to simulate a longer video."""
        segments = [
            CaptionedSegment(segment_id=1, start=0.0, end=10.0, description="Homepage with navigation bar and search"),
            CaptionedSegment(segment_id=2, start=10.0, end=20.0, description="Product listing page with filters"),
            CaptionedSegment(segment_id=3, start=20.0, end=30.0, description="Product detail page with images and price"),
            CaptionedSegment(segment_id=4, start=30.0, end=40.0, description="Shopping cart with 2 items"),
            CaptionedSegment(segment_id=5, start=40.0, end=55.0, description="Checkout form with payment fields"),
            CaptionedSegment(segment_id=6, start=55.0, end=65.0, description="Order confirmation page with order number"),
        ]
        sentences = [
            VoiceoverSentence(sentence_id=1, text="Let me show you our new e-commerce platform.", start=0.0, end=3.0),
            VoiceoverSentence(sentence_id=2, text="You can browse products and filter by category.", start=3.0, end=7.0),
            VoiceoverSentence(sentence_id=3, text="Each product has detailed images and pricing.", start=7.0, end=11.0),
            VoiceoverSentence(sentence_id=4, text="Add items to your cart with one click.", start=11.0, end=14.5),
            VoiceoverSentence(sentence_id=5, text="Checkout is fast and secure.", start=14.5, end=17.0),
            VoiceoverSentence(sentence_id=6, text="And you'll get a confirmation with your order number.", start=17.0, end=21.0),
        ]

        result = map_sentences_to_segments(segments, sentences, settings)

        print("\n--- 6-scene mapping ---")
        for entry in result:
            print(f"  Sentence {entry.sentence_id} → Segment {entry.segment_id} (speed: {entry.speed_factor}x)")

        assert len(result) == 6
        # Basic sanity: all sentences mapped, all segment IDs valid
        assert {e.sentence_id for e in result} == {1, 2, 3, 4, 5, 6}
        valid_ids = {s.segment_id for s in segments}
        assert all(e.segment_id in valid_ids for e in result)
        assert all(0.25 <= e.speed_factor <= 4.0 for e in result)
