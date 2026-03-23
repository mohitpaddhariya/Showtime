"""Frame captioning: OCR + structural analysis for richer keyframe descriptions.

Combines Tesseract OCR text with image structural features (colors, layout)
to give the AI mapper better context for matching narration to screen content.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from app.core.exceptions import CaptionError
from app.models.domain import CaptionedSegment, VideoSegment


def caption_segments(segments: list[VideoSegment]) -> list[CaptionedSegment]:
    """Run OCR + structural analysis on each segment's keyframe.

    Segments where keyframe_path is None get an empty description.
    """
    results: list[CaptionedSegment] = []

    for seg in segments:
        description = _describe_keyframe(seg.keyframe_path)
        results.append(
            CaptionedSegment(
                segment_id=seg.segment_id,
                start=seg.start,
                end=seg.end,
                is_idle=seg.is_idle,
                keyframe_path=seg.keyframe_path,
                description=description,
            )
        )

    return results


def _describe_keyframe(keyframe_path: Path | None) -> str:
    """Build a rich description combining OCR text and visual structure."""
    if keyframe_path is None or not keyframe_path.exists():
        return ""

    try:
        parts: list[str] = []

        # 1. OCR text
        ocr_text = _ocr_keyframe(keyframe_path)
        if ocr_text:
            parts.append(f"Text on screen: {ocr_text}")

        # 2. Structural analysis
        structure = _analyze_structure(keyframe_path)
        if structure:
            parts.append(f"Visual: {structure}")

        return " | ".join(parts) if parts else ""
    except Exception as e:
        raise CaptionError(f"Captioning failed on {keyframe_path}: {e}") from e


def _ocr_keyframe(keyframe_path: Path) -> str:
    """Run Tesseract OCR on a keyframe image."""
    image = Image.open(keyframe_path)
    text = pytesseract.image_to_string(image)
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return cleaned


def _analyze_structure(keyframe_path: Path) -> str:
    """Analyze visual structure of a keyframe for richer descriptions.

    Detects:
    - Dominant colors (suggests type of content)
    - UI element density (busy vs simple screen)
    - Presence of forms/inputs (rectangles in typical form locations)
    - Image regions (photos, charts, illustrations)
    """
    img = cv2.imread(str(keyframe_path))
    if img is None:
        return ""

    h, w = img.shape[:2]
    features: list[str] = []

    # Dominant color analysis
    color_desc = _describe_dominant_colors(img)
    if color_desc:
        features.append(color_desc)

    # Content density
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (h * w)

    if edge_density > 0.15:
        features.append("complex/busy layout")
    elif edge_density > 0.05:
        features.append("moderate layout")
    else:
        features.append("simple/minimal layout")

    # Detect rectangular regions (forms, buttons, cards)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_rects = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area_ratio = (cw * ch) / (w * h)
        if 0.01 < area_ratio < 0.5 and 0.2 < cw / max(ch, 1) < 5:
            large_rects += 1

    if large_rects > 10:
        features.append("many UI elements (form/list/table)")
    elif large_rects > 3:
        features.append("several UI elements (buttons/cards)")

    # Check for image-heavy regions (areas with high color variance)
    color_variance = np.std(img.astype(float), axis=(0, 1)).mean()
    if color_variance > 60:
        features.append("contains images or colorful graphics")

    return ", ".join(features) if features else ""


def _describe_dominant_colors(img: np.ndarray) -> str:
    """Describe the dominant colors in the image."""
    # Sample center region to avoid chrome/toolbars
    h, w = img.shape[:2]
    margin_h, margin_w = h // 10, w // 10
    center = img[margin_h:h - margin_h, margin_w:w - margin_w]

    if center.size == 0:
        return ""

    mean_color = center.mean(axis=(0, 1))  # BGR
    b, g, r = mean_color

    # Classify background color
    brightness = (r + g + b) / 3

    if brightness > 200:
        return "light/white background"
    elif brightness < 50:
        return "dark/black background"
    elif r > 150 and g < 100 and b < 100:
        return "red-dominant theme"
    elif g > 150 and r < 100 and b < 100:
        return "green-dominant theme"
    elif b > 150 and r < 100 and g < 100:
        return "blue-dominant theme"
    else:
        return ""
