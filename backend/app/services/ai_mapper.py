"""Vision-first AI mapping: match voiceover sentences to screen segments.

Architecture (v2 — single-call vision-first with built-in self-critique):

  1. ALL keyframes + full transcript -> ONE Llama 4 Scout vision call
     Model SEES actual screens and matches narration to visual content.
     Built-in self-critique: model rates its own pacing confidence (0-10).

  2. If pacing_score < threshold (default 8.5): ONE text refinement call
     The text model reviews the mapping's pacing analysis and adjusts.

  3. Fallback: chronological mapping if all AI fails (0 Groq calls).

Groq call budget: 1 call (vision mapping) + 0-1 call (refinement) = 1-2 total.
Previous v1 architecture could use up to 5 calls. This is 60-80% fewer API calls.

Token optimization:
- Keyframe images resized to 1024px max before base64 encoding
- All images batched in a single multimodal message
- Structured JSON mode enforced (no markdown/prose parsing needed)
- Total payload for 8 segments + 10 sentences: ~3500 tokens in, ~800 out
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import ollama
from groq import Groq

from app.core.config import Settings
from app.core.exceptions import MappingError
from app.models.domain import CaptionedSegment, MappingEntry, VoiceoverSentence
from app.services.vision_utils import build_image_content_blocks

logger = logging.getLogger(__name__)


# ── Vision Mapping Prompt (comprehensive single-call) ────────────────
# This prompt is the core of v2. It combines what used to be 3 separate
# prompts (vision, text, refinement) into ONE prompt with self-critique.

VISION_PROMPT = """\
You are an elite video editor creating a polished demo video from a screen recording.

I'm showing you ALL keyframe screenshots from the recording, plus the complete voiceover transcript.

YOUR TASK: Match each voiceover sentence to the screen segment that should be VISIBLE while the narrator speaks.

MAPPING STRATEGY (in priority order):
1. CONTENT MATCH: What the narrator describes should match what's on screen.
   Look at the actual visual content in each keyframe image.
2. TIMING: Earlier sentences generally map to earlier segments.
3. DURATION AWARENESS: Check that video segment duration roughly fits the audio.
   speed = video_duration / audio_duration
   Comfortable range: 0.5x - 2.0x. Outside this range = pacing problem.

PLAY vs FREEZE:
- PLAY (freeze=false): Video plays normally. Use for MOST sentences.
  Viewer sees the screen in motion — typing, scrolling, clicking.
- FREEZE (freeze=true): Hold one frame still. Use ONLY when ALL of these:
  * Screen shows a static result, table, or text block
  * Narrator is reading/explaining that specific static content
  * No action happening on screen during this segment
  Maximum 3 freeze moments in the entire video. Default to PLAY.

SELF-CRITIQUE (required):
After mapping, calculate speed for each: speed = video_duration / audio_duration
- All speeds 0.5-2.0x -> pacing_score = 9-10 (excellent)
- Some speeds outside range -> pacing_score = 6-8 (acceptable)
- Many extreme speeds -> pacing_score = 3-5 (needs refinement)

HARD RULES:
- CHRONOLOGICAL: segment_ids must be non-decreasing (1,1,2,3 OK. 1,3,2 NOT OK)
- Every sentence must be assigned exactly once
- speed_factor = 1.0 always (renderer handles actual speed adjustment)
- Maximum 3 freeze=true entries
- confidence: 0.0-1.0 per mapping (how certain is this match)

Return ONLY valid JSON:
{"mappings": [{"sentence_id": int, "segment_id": int, "speed_factor": 1.0, "freeze": bool, "confidence": float, "reasoning": "brief explanation"}, ...], "pacing_score": float, "pacing_notes": "brief summary"}"""


# ── Text Fallback Prompt (used when vision unavailable) ──────────────

TEXT_PROMPT = """\
You are a professional video editor syncing a voiceover to a screen recording.

For each sentence, decide:
1. Which segment it belongs to (match narration to content)
2. freeze: true ONLY for static screens being read/explained (max 3 in whole video). Default false.

CHRONOLOGICAL ORDER required: segment IDs must be non-decreasing.

Return ONLY valid JSON:
{"mappings": [{"sentence_id": int, "segment_id": int, "speed_factor": 1.0, "freeze": bool}, ...]}"""


# ── Refinement Prompt (text model, max 1 call) ──────────────────────

_REFINE_PROMPT = """\
You are reviewing a video edit timeline for pacing issues.

Current mapping with pacing analysis:
{analysis}

Available segments:
{seg_info}

FIX THESE ISSUES:
- Speed > 2.0x -> map to a longer segment or set freeze=true
- Speed < 0.5x -> map to a shorter segment or mark as freeze
- Long freeze > 8s -> switch to PLAY or remap to a different segment
- Maximum 3 freeze clips total
- Chronological order required (non-decreasing segment_ids)

If acceptable, respond: {{"action": "keep"}}
If you want to fix it: {{"action": "refine", "mappings": [{{"sentence_id": int, "segment_id": int, "speed_factor": 1.0, "freeze": bool}}, ...]}}"""


def map_sentences_to_segments(
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    settings: Settings | None = None,
) -> list[MappingEntry]:
    """Map each voiceover sentence to a screen segment.

    Strategy:
    1. Vision mapping (Llama 4 Scout, 1 call) — primary, sees actual keyframes
    2. Text mapping (Groq text / Ollama, 1 call) — fallback if no keyframes
    3. Chronological mapping (0 calls) — last resort

    If vision mapping's self-rated pacing_score < threshold, triggers ONE
    text-model refinement call to fix pacing issues. Max 2 total Groq calls.
    """
    if settings is None:
        settings = Settings()

    if not sentences:
        return []
    if not segments:
        raise MappingError("No screen segments provided for mapping")

    segments_by_id = {s.segment_id: s for s in segments}
    max_freeze = settings.max_freeze_count

    # ── Primary: Vision mapping (1 Groq vision call) ────────────
    has_keyframes = any(
        s.keyframe_path and Path(str(s.keyframe_path)).exists()
        for s in segments
    )

    if settings.groq_api_key and has_keyframes:
        try:
            result, pacing_score = _vision_mapping(segments, sentences, settings)
            result = _cap_freeze_count(result, max_freeze)
            if _is_chronological(result, segments_by_id):
                logger.info(
                    "Vision mapping accepted (pacing_score=%.1f, %d entries)",
                    pacing_score, len(result),
                )
                # Trigger refinement only if pacing is poor
                if pacing_score < settings.pacing_threshold:
                    logger.info("Pacing below %.1f, triggering refinement", settings.pacing_threshold)
                    try:
                        refined = _refine_once(
                            result, segments, sentences, segments_by_id, settings,
                        )
                        if refined is not None:
                            result = _cap_freeze_count(refined, max_freeze)
                            logger.info("Refinement accepted")
                    except Exception as e:
                        logger.warning("Refinement failed, keeping original: %s", e)
                return result
            else:
                logger.warning("Vision mapping rejected: non-chronological")
        except Exception as e:
            logger.warning("Vision mapping failed: %s", e)

    # ── Secondary: Text mapping (1 call) ────────────────────────
    try:
        if settings.llm_provider == "groq" and settings.groq_api_key:
            raw = _call_groq_text(segments, sentences, settings)
        elif settings.llm_provider == "ollama":
            raw = _call_ollama(segments, sentences, settings)
        else:
            raise MappingError("No AI provider configured")

        result = _parse_response(raw, segments, sentences)
        result = _cap_freeze_count(result, max_freeze)
        if _is_chronological(result, segments_by_id):
            logger.info("Text mapping accepted (%d entries)", len(result))
            return result
        else:
            logger.warning("Text mapping rejected: non-chronological")
    except Exception as e:
        logger.warning("Text mapping failed: %s", e)

    # ── Fallback: Chronological (0 calls) ───────────────────────
    logger.info("Using chronological fallback mapping")
    return _scene_aware_mapping(segments, sentences)


# ── Vision Mapping (primary strategy) ────────────────────────────────


_MAX_VISION_IMAGES = 5  # Groq Llama 4 Scout hard limit per request


def _vision_mapping(
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    settings: Settings,
) -> tuple[list[MappingEntry], float]:
    """Send keyframes + transcript to Llama 4 Scout for visual matching.

    Groq limits Llama 4 Scout to 5 images per request. When there are
    more than 5 segments, we pick the 5 most evenly-spaced keyframes
    as images and include the rest as text-only descriptions so the
    model still knows about every segment.

    Returns (mappings, pacing_score).
    """
    client = Groq(api_key=settings.groq_api_key)

    # When > 5 segments, select 5 evenly-spaced ones for images,
    # and provide text descriptions for the rest.
    if len(segments) > _MAX_VISION_IMAGES:
        # Pick evenly-spaced indices for the image slots
        step = len(segments) / _MAX_VISION_IMAGES
        image_indices = {int(i * step) for i in range(_MAX_VISION_IMAGES)}

        content: list[dict] = []
        for idx, seg in enumerate(segments):
            tag_info = f", tag={seg.semantic_tag}" if getattr(seg, "semantic_tag", None) else ""
            if idx in image_indices:
                # Include image for this segment
                content.extend(build_image_content_blocks([seg], max_dim=1024))
            else:
                # Text-only description (no image, stays under the limit)
                content.append({
                    "type": "text",
                    "text": (
                        f"\n--- Segment {seg.segment_id} "
                        f"(time: {seg.start:.1f}s - {seg.end:.1f}s, "
                        f"duration: {seg.duration:.1f}s{tag_info}) ---\n"
                        f"[Text only] {seg.description[:200]}"
                    ),
                })
    else:
        content = build_image_content_blocks(segments, max_dim=1024)

    # Append transcript with timing info
    transcript_text = "\n\nVOICEOVER TRANSCRIPT:\n"
    for s in sentences:
        transcript_text += (
            f"  Sentence {s.sentence_id} ({s.start:.1f}s-{s.end:.1f}s, "
            f"duration={s.duration:.1f}s): \"{s.text}\"\n"
        )
    transcript_text += (
        "\nMatch each sentence to the segment whose VISUAL CONTENT best matches "
        "the narration. Rate your pacing confidence 0-10. Return JSON only."
    )
    content.append({"type": "text", "text": transcript_text})

    response = client.chat.completions.create(
        model=settings.groq_vision_model,
        messages=[
            {"role": "system", "content": VISION_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    logger.debug("Vision response: %s", raw[:500])

    # Extract pacing_score before parsing mappings
    data = json.loads(raw)
    pacing_score = float(data.get("pacing_score", 10.0))

    entries = _parse_response(raw, segments, sentences)
    return entries, pacing_score


# ── Refinement (max 1 text call) ─────────────────────────────────────


def _refine_once(
    current: list[MappingEntry],
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    segments_by_id: dict,
    settings: Settings,
) -> list[MappingEntry] | None:
    """One-shot text-model refinement. Returns None if AI says "keep".

    Only called when the vision model's pacing_score < threshold.
    Uses the cheaper/faster text model (Llama 3.3 70B) since it only
    needs to review numbers and segment metadata, not images.
    """
    if not settings.groq_api_key:
        return None

    sentences_by_id = {s.sentence_id: s for s in sentences}
    analysis = _analyze_mapping(current, segments_by_id, sentences_by_id)

    seg_info = "\n".join(
        f"Seg {s.segment_id}: {s.start:.1f}-{s.end:.1f}s ({s.duration:.1f}s) | {s.description[:100]}"
        for s in segments
    )

    prompt = _REFINE_PROMPT.format(analysis=analysis, seg_info=seg_info)

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Review this mapping and respond with JSON:\n\n{analysis}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    if data.get("action") == "keep":
        return None

    refined = _parse_response(raw, segments, sentences)
    if not _is_chronological(refined, segments_by_id):
        logger.warning("Refinement rejected: non-chronological")
        return None

    # Accept only if it actually reduces warning count
    old_warnings = analysis.count("WARNING")
    new_analysis = _analyze_mapping(refined, segments_by_id, sentences_by_id)
    new_warnings = new_analysis.count("WARNING")

    if new_warnings < old_warnings:
        logger.info("Refinement improved: %d -> %d warnings", old_warnings, new_warnings)
        return refined

    logger.info("Refinement did not improve (%d warnings), keeping original", old_warnings)
    return None


def _analyze_mapping(
    mappings: list[MappingEntry],
    segments_by_id: dict,
    sentences_by_id: dict,
) -> str:
    """Build a human-readable pacing analysis for AI review.

    Flags issues like extreme speeds, long freezes, mismatched content.
    """
    lines = []
    for m in mappings:
        seg = segments_by_id.get(m.segment_id)
        sent = sentences_by_id.get(m.sentence_id)
        if not seg or not sent:
            continue

        vid_dur = seg.end - seg.start
        aud_dur = sent.duration
        speed = vid_dur / aud_dur if aud_dur > 0 else 0

        mode = "FREEZE" if m.freeze else "PLAY"
        line = (
            f"Sen {m.sentence_id} -> Seg {m.segment_id} [{mode}] | "
            f"video={vid_dur:.1f}s audio={aud_dur:.1f}s speed={speed:.2f}x"
        )

        if speed > 2.0:
            line += " WARNING: video too long for sentence (will play too fast)"
        elif speed < 0.5:
            line += " WARNING: video too short (will play very slow)"
        elif m.freeze and aud_dur > 8.0:
            line += " WARNING: long freeze (>8s)"
        if vid_dur < 1.5 and not m.freeze:
            line += " WARNING: clip too short (<1.5s)"

        lines.append(line)

    return "\n".join(lines)


# ── Text Mapping (fallback when no keyframes) ───────────────────────


def _call_groq_text(segments, sentences, settings) -> str:
    """Text-only mapping via Groq. Uses segment descriptions instead of images."""
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": TEXT_PROMPT},
            {"role": "user", "content": _build_text_prompt(segments, sentences)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return response.choices[0].message.content


def _call_ollama(segments, sentences, settings) -> str:
    """Text-only mapping via local Ollama (offline, no API key needed)."""
    response = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": TEXT_PROMPT},
            {"role": "user", "content": _build_text_prompt(segments, sentences)},
        ],
        format="json",
    )
    return response["message"]["content"]


def _build_text_prompt(segments, sentences) -> str:
    """Build the text context for non-vision mapping."""
    segments_data = [
        {"segment_id": s.segment_id, "start": s.start, "end": s.end,
         "duration": round(s.duration, 1), "description": s.description[:200]}
        for s in segments
    ]
    sentences_data = [
        {"sentence_id": s.sentence_id, "text": s.text,
         "start": round(s.start, 1), "end": round(s.end, 1),
         "duration": round(s.duration, 1)}
        for s in sentences
    ]
    return (
        f"Screen segments (with timestamps and descriptions):\n{json.dumps(segments_data, indent=2)}\n\n"
        f"Voiceover sentences (with timestamps):\n{json.dumps(sentences_data, indent=2)}\n\n"
        f"Match each sentence to the segment whose content best matches the narration. "
        f"Use timestamps as a secondary signal — earlier sentences tend to map to earlier segments."
    )


# ── Chronological Fallback (0 API calls) ────────────────────────────


def _scene_aware_mapping(segments, sentences) -> list[MappingEntry]:
    """Map by time scaling: sentence midpoint -> proportional video position.

    This is the zero-API-call fallback when all AI strategies fail.
    """
    sorted_segs = sorted(segments, key=lambda s: s.start)
    total_video = sorted_segs[-1].end - sorted_segs[0].start
    total_audio = sentences[-1].end - sentences[0].start if sentences else 0

    if total_audio <= 0 or total_video <= 0:
        return [
            MappingEntry(sentence_id=s.sentence_id, segment_id=sorted_segs[0].segment_id)
            for s in sentences
        ]

    scale = total_video / total_audio
    video_offset = sorted_segs[0].start
    audio_offset = sentences[0].start

    entries = []
    for sentence in sentences:
        sent_mid = sentence.start + sentence.duration / 2
        video_time = video_offset + (sent_mid - audio_offset) * scale
        video_time = max(sorted_segs[0].start, min(sorted_segs[-1].end, video_time))
        best_seg = _find_segment_at_time(video_time, sorted_segs)
        entries.append(MappingEntry(
            sentence_id=sentence.sentence_id,
            segment_id=best_seg.segment_id,
            speed_factor=1.0,
        ))
    return entries


# ── Shared Utilities ────────────────────────────────────────────────


def _find_segment_at_time(time, sorted_segments):
    """Find the segment that contains the given time point."""
    for seg in sorted_segments:
        if seg.start <= time <= seg.end:
            return seg
    return min(sorted_segments, key=lambda s: abs((s.start + s.end) / 2 - time))


def _cap_freeze_count(entries: list[MappingEntry], max_freeze: int = 3) -> list[MappingEntry]:
    """Cap the number of freeze clips to prevent slideshow output.

    If too many freezes, keep only the last N (later sentences are more
    likely to show results/summaries worth freezing on).
    """
    freeze_count = sum(1 for e in entries if e.freeze)
    if freeze_count <= max_freeze:
        return entries

    logger.warning("LLM froze %d/%d sentences, capping to %d", freeze_count, len(entries), max_freeze)

    freeze_entries = [(i, e) for i, e in enumerate(entries) if e.freeze]
    keep_indices = {i for i, _ in freeze_entries[-max_freeze:]}

    result = []
    for i, e in enumerate(entries):
        if e.freeze and i not in keep_indices:
            result.append(MappingEntry(
                sentence_id=e.sentence_id,
                segment_id=e.segment_id,
                speed_factor=e.speed_factor,
                freeze=False,
                confidence=e.confidence,
                reasoning=e.reasoning,
            ))
        else:
            result.append(e)
    return result


def _parse_response(raw, segments, sentences) -> list[MappingEntry]:
    """Parse AI response JSON into MappingEntry list.

    Handles both old format (array or {"mappings": [...]}) and new format
    with confidence/reasoning fields. Backward compatible with all tests.
    """
    data = json.loads(raw)

    if isinstance(data, dict):
        for key in ("mappings", "mapping", "result", "results", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise MappingError(f"Unexpected JSON structure: {list(data.keys())}")

    if not isinstance(data, list):
        raise MappingError(f"Expected array, got {type(data).__name__}")

    valid_seg_ids = {s.segment_id for s in segments}
    expected_sent_ids = {s.sentence_id for s in sentences}
    entries = []
    seen = set()

    for item in data:
        entry = MappingEntry(
            sentence_id=item["sentence_id"],
            segment_id=item["segment_id"],
            speed_factor=item.get("speed_factor", 1.0),
            freeze=item.get("freeze", False),
            confidence=item.get("confidence", 1.0),
            reasoning=item.get("reasoning", ""),
        )
        if entry.segment_id not in valid_seg_ids:
            raise MappingError(f"Invalid segment_id {entry.segment_id}")
        seen.add(entry.sentence_id)
        entries.append(entry)

    missing = expected_sent_ids - seen
    if missing:
        raise MappingError(f"Missing sentence_ids: {missing}")

    entries.sort(key=lambda e: e.sentence_id)
    return entries


def _is_chronological(mappings, segments_by_id) -> bool:
    """Check that segment starts are non-decreasing (no backward video jumps)."""
    prev_start = -1.0
    for m in mappings:
        seg = segments_by_id.get(m.segment_id)
        if seg is None:
            return False
        if seg.start < prev_start:
            return False
        prev_start = seg.start
    return True


# ── Backward-compat aliases (used by tests and external code) ───────


def _fallback_time_mapping(segments, sentences):
    """Alias for _scene_aware_mapping (test compatibility)."""
    return _scene_aware_mapping(segments, sentences)


def _find_best_overlap(sentence, segments):
    """Find the segment with maximum time overlap to a sentence (test compat)."""
    best = segments[0]
    best_overlap = -1.0
    for seg in segments:
        overlap = max(0.0, min(sentence.end, seg.end) - max(sentence.start, seg.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = seg
    if best_overlap == 0.0:
        mid = (sentence.start + sentence.end) / 2
        best = min(segments, key=lambda s: abs((s.start + s.end) / 2 - mid))
    return best


def _call_groq(segments, sentences, settings):
    """Alias for _call_groq_text (test compatibility)."""
    return _call_groq_text(segments, sentences, settings)
