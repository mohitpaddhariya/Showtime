"""Map voiceover sentences to screen segments using visual intelligence.

Pipeline (from the GPT approach — adapted for Groq):
1. Keyframes already extracted by scene detector (1 per scene)
2. Transcript already extracted by audio analyzer (word-level timestamps)
3. Send keyframe IMAGES + structured transcript to Llama 4 Scout (vision model on Groq)
4. LLM literally SEES the screens and decides where each sentence belongs
5. Fallback: text-only mapping with Llama 3.3 70B, then scene-aware chronological

All strategies enforce chronological order (no backward video jumps).
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import ollama
from groq import Groq

from app.core.config import Settings
from app.core.exceptions import MappingError
from app.models.domain import CaptionedSegment, MappingEntry, VoiceoverSentence

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2

VISION_PROMPT = """You are a professional video editor creating a polished demo video.
I'm showing you keyframe screenshots from a screen recording + a voiceover transcript.

For each sentence, decide:
1. Which screen segment it belongs to (match narration to visual content)
2. Whether to FREEZE or PLAY the video during that sentence

MATCHING STRATEGY:
- Match by CONTENT FIRST: what the narrator says should match what's visible on screen.
- Use TIMING as a secondary signal: sentences early in the voiceover likely map to early segments.
- Consider segment DURATION: prefer segments long enough to comfortably play during the sentence.

PLAY (default) = video plays normally. Use this for MOST sentences.
  The viewer sees the screen recording in motion — typing, scrolling, clicking, navigating.

FREEZE (rare) = hold one frame still. ONLY use when ALL of these are true:
  - The screen shows a static result, table, or text block
  - The narrator is specifically reading or explaining that static content
  - There is NO action happening on screen during this part of the recording
  Use sparingly — maximum 2-3 freeze moments in a whole video.

IMPORTANT: Default to PLAY. A video that's mostly frozen looks like a slideshow.

RULES:
- CHRONOLOGICAL ORDER: segment IDs must be non-decreasing (1,1,2,3 OK. 1,3,2 NOT OK).
- Multiple sentences can share the same segment.
- Every sentence must be assigned.
- speed_factor = 1.0 always.
- freeze = false for most sentences. Only true for 2-3 max.

Return ONLY valid JSON:
{"mappings": [{"sentence_id": int, "segment_id": int, "speed_factor": 1.0, "freeze": bool}, ...]}"""

TEXT_PROMPT = """You are a professional video editor syncing a voiceover to a screen recording.

For each sentence, decide:
1. Which segment it belongs to (match narration to content)
2. freeze: true ONLY for static screens being read/explained (max 2-3 in whole video). Default false.

CHRONOLOGICAL ORDER required: segment IDs must be non-decreasing.

Return ONLY valid JSON:
{"mappings": [{"sentence_id": int, "segment_id": int, "speed_factor": 1.0, "freeze": bool}, ...]}"""


_MAX_REFINE_PASSES = 2  # max refinement iterations (not infinite)


def map_sentences_to_segments(
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    settings: Settings | None = None,
) -> list[MappingEntry]:
    """Map each voiceover sentence to a screen segment.

    Pipeline:
    1. Initial mapping (vision or text or fallback)
    2. Refinement loop (max 2 passes) — AI reviews its own mapping,
       sees the pacing issues, and adjusts. Passes previous context
       so it doesn't repeat mistakes.
    """
    if settings is None:
        settings = Settings()

    if not sentences:
        return []
    if not segments:
        raise MappingError("No screen segments provided for mapping")

    segments_by_id = {s.segment_id: s for s in segments}

    # ── Step 1: Initial mapping ───────────────────────────────
    result = _initial_mapping(segments, sentences, segments_by_id, settings)

    # ── Step 2: Refinement loop ───────────────────────────────
    result = _refinement_loop(result, segments, sentences, segments_by_id, settings)

    return result


def _initial_mapping(
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    segments_by_id: dict,
    settings: Settings,
) -> list[MappingEntry]:
    """Get the first mapping — vision, text, or fallback."""
    # Try vision model first
    if settings.llm_provider == "groq" and settings.groq_api_key:
        has_keyframes = any(s.keyframe_path and s.keyframe_path.exists() for s in segments)
        if has_keyframes:
            try:
                result = _map_with_vision(segments, sentences, settings)
                result = _cap_freeze_count(result)
                if _is_chronological(result, segments_by_id):
                    logger.info("Vision mapping accepted (Llama 4 Scout)")
                    return result
                else:
                    logger.warning("Vision mapping rejected: non-chronological")
            except Exception as e:
                logger.warning("Vision mapping failed: %s", e)

    # Try text-only LLM
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            if settings.llm_provider == "groq":
                raw = _call_groq_text(segments, sentences, settings)
            else:
                raw = _call_ollama(segments, sentences, settings)

            result = _parse_response(raw, segments, sentences)
            result = _cap_freeze_count(result)
            if _is_chronological(result, segments_by_id):
                logger.info("Text mapping accepted")
                return result
            else:
                logger.warning("Text mapping rejected: non-chronological")
                break
        except Exception as e:
            logger.warning("Text mapping attempt %d failed: %s", attempt, e)

    # Scene-aware chronological fallback
    logger.info("Using scene-aware chronological fallback")
    return _scene_aware_mapping(segments, sentences)


def _refinement_loop(
    current: list[MappingEntry],
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    segments_by_id: dict,
    settings: Settings,
) -> list[MappingEntry]:
    """Self-review loop: AI sees its mapping + pacing stats, decides to refine or keep.

    Max `_MAX_REFINE_PASSES` iterations. Passes full previous context so it
    doesn't repeat the same mistakes. Stops early if AI says "keep".
    """
    if not settings.groq_api_key:
        return current  # can't refine without API

    sentences_by_id = {s.sentence_id: s for s in sentences}
    best = current

    for pass_num in range(1, _MAX_REFINE_PASSES + 1):
        try:
            # Build pacing analysis of current mapping
            analysis = _analyze_mapping(best, segments_by_id, sentences_by_id)

            # Ask AI to review
            refined = _call_refinement(best, segments, sentences, analysis, pass_num, settings)

            if refined is None:
                logger.info("Refinement pass %d: AI says keep current mapping", pass_num)
                break

            refined = _cap_freeze_count(refined)

            if not _is_chronological(refined, segments_by_id):
                logger.warning("Refinement pass %d rejected: non-chronological", pass_num)
                break

            # Check if refinement actually improved anything
            old_issues = analysis.count("WARNING")
            new_analysis = _analyze_mapping(refined, segments_by_id, sentences_by_id)
            new_issues = new_analysis.count("WARNING")

            if new_issues < old_issues:
                logger.info("Refinement pass %d accepted: %d → %d issues", pass_num, old_issues, new_issues)
                best = refined
            else:
                logger.info("Refinement pass %d: no improvement (%d issues), keeping previous", pass_num, old_issues)
                break

        except Exception as e:
            logger.warning("Refinement pass %d failed: %s", pass_num, e)
            break

    return best


def _analyze_mapping(
    mappings: list[MappingEntry],
    segments_by_id: dict,
    sentences_by_id: dict,
) -> str:
    """Build a human-readable pacing analysis of the current mapping.

    Flags issues like extreme speeds, long freezes, mismatched content.
    This gets passed to the AI so it can see what needs fixing.
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
        line = f"Sen {m.sentence_id} → Seg {m.segment_id} [{mode}] | video={vid_dur:.1f}s audio={aud_dur:.1f}s speed={speed:.2f}x"

        # Flag pacing issues
        if speed > 2.0:
            line += " ⚠ WARNING: video way too long for this sentence (will play very fast, choppy)"
        elif speed < 0.5:
            line += " ⚠ WARNING: video too short for this sentence (will play very slow/freeze)"
        elif m.freeze and aud_dur > 8.0:
            line += " ⚠ WARNING: long freeze (>8s) — viewer may get bored"
        if vid_dur < 1.5 and not m.freeze:
            line += " ⚠ WARNING: video clip too short (<1.5s) — consider mapping to a longer segment"

        lines.append(line)

    return "\n".join(lines)


REFINE_PROMPT = """You are reviewing your own video edit mapping. Here is what you mapped previously, along with pacing analysis:

{analysis}

ISSUES TO LOOK FOR:
- Any WARNING lines need fixing
- Sentences with speed > 2.0x will look too fast and choppy — try mapping them to a longer segment
- Sentences with speed < 0.5x will look frozen — consider marking as freeze=true or remapping
- Video clips shorter than 1.5s will look bad — try mapping to a longer segment
- Long freezes (>8s) bore the viewer — switch to PLAY or split across segments
- If everything looks good, respond with: {{"action": "keep"}}

If you want to adjust, respond with:
{{"action": "refine", "mappings": [{{"sentence_id": int, "segment_id": int, "speed_factor": 1.0, "freeze": bool}}, ...]}}

RULES: Chronological order (non-decreasing segment IDs). Max 3 freezes total."""


def _call_refinement(
    current: list[MappingEntry],
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    analysis: str,
    pass_num: int,
    settings: Settings,
) -> list[MappingEntry] | None:
    """Ask the AI to review its mapping and optionally refine.

    Returns None if AI says "keep", or new mappings if it wants to refine.
    """
    if not settings.groq_api_key:
        return None

    # Build context with segments info for the AI
    seg_info = "\n".join(
        f"Seg {s.segment_id}: {s.start:.1f}-{s.end:.1f}s ({s.duration:.1f}s) | {s.description[:100]}"
        for s in segments
    )

    prompt = REFINE_PROMPT.format(analysis=analysis)
    user_msg = f"Pass {pass_num} review. Respond with json.\n\nAvailable segments:\n{seg_info}\n\nCurrent mapping:\n{analysis}"

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    # Check if AI wants to keep current mapping
    action = data.get("action", "keep")
    if action == "keep":
        return None

    # Parse refined mappings
    return _parse_response(raw, segments, sentences)


# ── Strategy 1: Vision Mapping (Llama 4 Scout) ───────────────────────


def _map_with_vision(
    segments: list[CaptionedSegment],
    sentences: list[VoiceoverSentence],
    settings: Settings,
) -> list[MappingEntry]:
    """Send keyframe images + transcript to Llama 4 Scout for visual matching.

    The model SEES each screen and matches narration to the right visual moment.
    """
    client = Groq(api_key=settings.groq_api_key)

    # Build multimodal message content
    content = []

    # Add each segment with its keyframe image
    for seg in segments:
        content.append({
            "type": "text",
            "text": f"\n--- Segment {seg.segment_id} (time: {seg.start:.1f}s - {seg.end:.1f}s, duration: {seg.duration:.1f}s) ---",
        })

        if seg.keyframe_path and seg.keyframe_path.exists():
            img_bytes = seg.keyframe_path.read_bytes()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        else:
            content.append({
                "type": "text",
                "text": f"[No image. OCR: {seg.description[:150]}]",
            })

    # Add transcript
    transcript_text = "\n\nVOICEOVER TRANSCRIPT:\n"
    for s in sentences:
        transcript_text += f"  Sentence {s.sentence_id} ({s.start:.1f}s-{s.end:.1f}s): \"{s.text}\"\n"
    transcript_text += "\nAssign each sentence to the segment whose VISUAL CONTENT matches the narration. Return JSON only."

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
    return _parse_response(raw, segments, sentences)


# ── Strategy 2: Text-only Mapping ─────────────────────────────────────


def _call_groq_text(segments, sentences, settings) -> str:
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


# ── Strategy 3: Scene-aware Fallback ──────────────────────────────────


def _scene_aware_mapping(segments, sentences) -> list[MappingEntry]:
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


# ── Shared Utilities ──────────────────────────────────────────────────


def _find_segment_at_time(time, sorted_segments):
    for seg in sorted_segments:
        if seg.start <= time <= seg.end:
            return seg
    return min(sorted_segments, key=lambda s: abs((s.start + s.end) / 2 - time))


def _cap_freeze_count(entries: list[MappingEntry], max_freeze: int = 3) -> list[MappingEntry]:
    """Cap the number of freeze clips to prevent slideshow output.

    If the LLM marks too many as freeze, keep only the longest sentences
    as freeze (those are most likely reading moments) and set the rest to PLAY.
    """
    freeze_count = sum(1 for e in entries if e.freeze)
    if freeze_count <= max_freeze:
        return entries

    logger.warning("LLM froze %d/%d sentences — capping to %d", freeze_count, len(entries), max_freeze)

    # Keep freeze only on the longest-duration freeze entries
    freeze_entries = [(i, e) for i, e in enumerate(entries) if e.freeze]
    # We don't have sentence duration here, so keep the last N freezes
    # (later sentences are more likely to be results/summaries worth freezing)
    keep_indices = {i for i, _ in freeze_entries[-max_freeze:]}

    result = []
    for i, e in enumerate(entries):
        if e.freeze and i not in keep_indices:
            result.append(MappingEntry(
                sentence_id=e.sentence_id,
                segment_id=e.segment_id,
                speed_factor=e.speed_factor,
                freeze=False,
            ))
        else:
            result.append(e)
    return result


def _parse_response(raw, segments, sentences) -> list[MappingEntry]:
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
    prev_start = -1.0
    for m in mappings:
        seg = segments_by_id.get(m.segment_id)
        if seg is None:
            return False
        if seg.start < prev_start:
            return False
        prev_start = seg.start
    return True


# ── Compat aliases for tests ─────────────────────────────────────────

def _fallback_time_mapping(segments, sentences):
    return _scene_aware_mapping(segments, sentences)

def _find_best_overlap(sentence, segments):
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
    """Compat for live tests."""
    return _call_groq_text(segments, sentences, settings)
