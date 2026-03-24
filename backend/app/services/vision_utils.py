"""Shared vision utilities for keyframe image processing.

Handles:
- Keyframe resizing to reduce Groq payload size
- Base64 encoding for multimodal API calls
- Building image content blocks for batched vision requests

Token/cost optimization notes for Groq free tier:
- Each base64 image adds ~1.3x the raw bytes to the prompt token count
- A 1920x1080 PNG screenshot is ~200-400KB raw, ~300-550KB base64
- Resizing to max 1024px keeps each image under ~200KB base64
- Batching 4-8 images in ONE call stays under Groq's 4MB payload limit
- Groq free tier: 30 req/min, 14400 req/day — every saved call matters
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np


def resize_keyframe(keyframe_path: Path, max_dim: int = 1024) -> bytes:
    """Read and resize a keyframe image, returning PNG bytes.

    Shrinks the longest dimension to max_dim while preserving aspect ratio.
    A 1920x1080 screenshot becomes ~1024x576 (~4x fewer pixels), which
    dramatically reduces the base64 payload for Groq API calls.
    """
    img = cv2.imread(str(keyframe_path))
    if img is None:
        return keyframe_path.read_bytes()

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def encode_keyframe_b64(keyframe_path: Path, max_dim: int = 1024) -> str:
    """Read, resize, and base64-encode a keyframe for API transmission."""
    img_bytes = resize_keyframe(keyframe_path, max_dim)
    return base64.b64encode(img_bytes).decode("utf-8")


def build_image_content_blocks(
    segments: list,
    max_dim: int = 1024,
) -> list[dict]:
    """Build multimodal content blocks for a batch of segment keyframes.

    Returns a list of content items suitable for Groq's chat API:
    - Text block with segment metadata (timing, duration)
    - Image block with base64-encoded resized keyframe

    All images are batched into a SINGLE list so they can be sent in ONE
    API call — this is the key optimization over sending one image per call.

    Payload estimate for 8 segments: ~1.5MB total (well under Groq's 4MB limit).
    """
    content: list[dict] = []

    for seg in segments:
        # Segment metadata text block
        tag_info = f", tag={seg.semantic_tag}" if getattr(seg, "semantic_tag", None) else ""
        content.append({
            "type": "text",
            "text": (
                f"\n--- Segment {seg.segment_id} "
                f"(time: {seg.start:.1f}s - {seg.end:.1f}s, "
                f"duration: {seg.duration:.1f}s{tag_info}) ---"
            ),
        })

        # Keyframe image block (resized to reduce payload)
        kf_path = getattr(seg, "keyframe_path", None)
        if kf_path and Path(str(kf_path)).exists():
            b64 = encode_keyframe_b64(Path(str(kf_path)), max_dim)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        else:
            # No image available — include text description as fallback
            desc = getattr(seg, "description", "")
            content.append({
                "type": "text",
                "text": f"[No image available. Description: {desc[:200] if desc else 'none'}]",
            })

    return content
