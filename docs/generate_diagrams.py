#!/usr/bin/env python3
"""Generate clean, projector-friendly Showtime Pipeline diagrams (v2).

Design rules:
  - Minimum 14pt text (16-18 preferred)
  - Maximum white space
  - One clear message per diagram
  - Strong color hierarchy
  - No clutter
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# ── BRAND ───────────────────────────────────────────────────
NAVY      = "#003087"
ACCENT    = "#0072CE"
SKY       = "#D6E4F0"
ICE       = "#EEF2F7"
GREEN     = "#0F7B3F"
GREEN_LT  = "#D4EDDA"
ORANGE    = "#C87A1A"
ORANGE_LT = "#FFF3E0"
RED       = "#C0392B"
RED_LT    = "#FDEDEC"
GRAY      = "#8E99A4"
GRAY_LT   = "#CED4DA"
DARK      = "#212529"
WHITE     = "#FFFFFF"
PURPLE    = "#6F42C1"
PURPLE_LT = "#F0E6FF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.unicode_minus": False,
})


def save(fig, name, dpi=300):
    fig.savefig(OUT / f"{name}.png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none", pad_inches=0.4)
    plt.close(fig)
    print(f"  {name}.png")


# ── HELPERS ─────────────────────────────────────────────────
def rounded_box(ax, x, y, w, h, text, sub=None, fc=SKY, ec=NAVY,
                tc=NAVY, fs=16, sub_fs=11, lw=2, pad=0.18, shadow=True):
    if shadow:
        s = FancyBboxPatch((x + 0.1, y - 0.1), w, h, boxstyle=f"round,pad={pad}",
                           fc=DARK, alpha=0.1, ec="none", zorder=1)
        ax.add_patch(s)

    r = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(r)
    if sub:
        ax.text(x + w/2, y + h * 0.65, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
        ax.text(x + w/2, y + h * 0.3, sub, ha="center", va="center",
                fontsize=sub_fs, color=GRAY, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)


def arrow_v(ax, x, y_from, y_to, color=NAVY, lw=2.5, text=None):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7", color=color, lw=lw), zorder=1)
    if text:
        ax.text(x + 0.2, (y_from + y_to)/2, text, fontsize=12, fontweight="bold", color=color, va="center")


def arrow_h(ax, x_from, y, x_to, color=NAVY, lw=2.5, text=None):
    ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7", color=color, lw=lw), zorder=1)
    if text:
        ax.text((x_from + x_to)/2, y + 0.2, text, fontsize=12, fontweight="bold", color=color, ha="center")


# ════════════════════════════════════════════════════════════
# 1. PIPELINE OVERVIEW (v2)
# ════════════════════════════════════════════════════════════
def diagram_pipeline():
    fig, ax = plt.subplots(figsize=(13, 13))
    fig.set_facecolor(WHITE)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # Title
    ax.text(6.5, 12.3, "Showtime Pipeline Architecture (v2)", ha="center",
            fontsize=26, fontweight="bold", color=NAVY)
    ax.text(6.5, 11.8, "3-4 Groq calls total per video", ha="center",
            fontsize=14, color=GRAY)

    # Inputs
    rounded_box(ax, 1.0, 10.0, 4.0, 1.2, "Raw Screen Recording", "Video (.mp4)",
                fc=ICE, ec=GRAY, tc=DARK, fs=18)
    rounded_box(ax, 8.0, 10.0, 4.0, 1.2, "Voiceover Audio", "Audio (.mp3)",
                fc=ICE, ec=GRAY, tc=DARK, fs=18)

    arrow_v(ax, 3.0, 10.0, 9.1)
    arrow_v(ax, 10.0, 10.0, 9.1)

    # Step 1: Scene Detection
    rounded_box(ax, 1.0, 7.7, 4.0, 1.4, "1. Scene Detection", "OpenCV: Diff & Split",
                fc=SKY, ec=ACCENT, tc=NAVY, fs=18)

    # Step 3: Transcription
    rounded_box(ax, 8.0, 7.7, 4.0, 1.4, "3. Transcription", "Groq Whisper (1 call)",
                fc=SKY, ec=ACCENT, tc=NAVY, fs=18)

    arrow_v(ax, 3.0, 7.7, 6.8)

    # Step 1b: AI Verification (NEW in v2)
    rounded_box(ax, 0.5, 5.4, 5.0, 1.4, "1b. AI Verification", "Llama 4 Scout (1 call, all keyframes)",
                fc=PURPLE_LT, ec=PURPLE, tc=PURPLE, fs=17, sub_fs=12)
    # v2 badge
    badge = FancyBboxPatch((4.8, 6.4), 0.9, 0.45, boxstyle="round,pad=0.08",
                            fc=PURPLE, ec=PURPLE, lw=0, zorder=5)
    ax.add_patch(badge)
    ax.text(5.25, 6.63, "NEW", ha="center", va="center",
            fontsize=10, fontweight="bold", color=WHITE, zorder=6)

    arrow_v(ax, 3.0, 5.4, 4.6)

    # Step 2: Frame Captioning
    rounded_box(ax, 1.0, 3.2, 4.0, 1.4, "2. Frame Captioning", "Tesseract OCR (local)",
                fc=SKY, ec=ACCENT, tc=NAVY, fs=18)

    # Connect Step 2 and 3 into AI Mapper
    ax.plot([3.0, 3.0], [3.2, 2.5], color=NAVY, lw=2.5, zorder=1)
    ax.plot([3.0, 6.5], [2.5, 2.5], color=NAVY, lw=2.5, zorder=1)

    # Transcription feeds down
    ax.plot([10.0, 10.0], [7.7, 2.5], color=NAVY, lw=2.5, zorder=1)
    ax.plot([10.0, 6.5], [2.5, 2.5], color=NAVY, lw=2.5, zorder=1)

    arrow_v(ax, 6.5, 2.5, 1.9, color=NAVY, lw=2.5)

    # Step 4: AI Mapping (v2)
    rounded_box(ax, 3.0, 0.3, 7.0, 1.6, "4. AI Mapping (Vision-First)",
                "1 Llama 4 Scout call + self-critique",
                fc=ORANGE_LT, ec=ORANGE, tc=ORANGE, fs=20, sub_fs=13, lw=3)

    # CORE AI badge
    badge2 = FancyBboxPatch((9.2, 1.5), 1.5, 0.5, boxstyle="round,pad=0.08",
                            fc=ORANGE, ec=ORANGE, lw=0, zorder=5)
    ax.add_patch(badge2)
    ax.text(9.95, 1.75, "CORE AI", ha="center", va="center",
            fontsize=11, fontweight="bold", color=WHITE, zorder=6)

    save(fig, "01_pipeline_overview")


# ════════════════════════════════════════════════════════════
# 2. AI MAPPER — Self-Critique + Optional Refinement (v2)
# ════════════════════════════════════════════════════════════
def diagram_ai_mapper():
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.set_facecolor(WHITE)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7, 8.3, "AI Mapping: Self-Critique + Refinement (v2)", ha="center", va="center",
            fontsize=24, fontweight="bold", color=NAVY)

    # Step A: Single Vision Call
    rounded_box(ax, 0.5, 4.5, 3.5, 2.0, "Vision Mapping", "1 call: ALL keyframes\n+ full transcript",
                fc=SKY, ec=NAVY, tc=NAVY, fs=18, sub_fs=12)

    arrow_h(ax, 4.0, 5.5, 5.5, NAVY)
    ax.text(4.75, 5.8, "JSON + score", ha="center", va="center", fontsize=11, fontweight="bold", color=DARK)

    # Step B: Self-Critique Check
    rounded_box(ax, 5.5, 4.5, 3.0, 2.0, "Self-Critique", "pacing_score\n0-10",
                fc=ICE, ec=GRAY, tc=DARK, fs=18, sub_fs=13)

    # Score >= 8.5: Accept
    arrow_h(ax, 8.5, 5.8, 10.5, GREEN)
    ax.text(9.5, 6.1, "Score >= 8.5", ha="center", va="center", fontsize=13, fontweight="bold", color=GREEN)

    # Score < 8.5: Refine
    arrow_v(ax, 7.0, 4.5, 3.5, RED, 2.5)
    ax.text(7.5, 4.0, "Score < 8.5", ha="left", va="center", fontsize=13, fontweight="bold", color=RED)

    # Refinement box
    rounded_box(ax, 5.5, 2.0, 3.0, 1.5, "Text Refinement", "Llama 3.3 (1 call)",
                fc=RED_LT, ec=RED, tc=RED, fs=16, sub_fs=12)

    arrow_h(ax, 8.5, 2.75, 10.5, GREEN)

    # Result
    rounded_box(ax, 10.5, 4.0, 3.0, 2.0, "Final Mapping", "Chronological EDL\nwith confidence",
                fc=GREEN_LT, ec=GREEN, tc=GREEN, fs=18, sub_fs=12)

    # Explanation
    desc = FancyBboxPatch((0.5, 0.0), 13.0, 1.5, boxstyle="round,pad=0.15",
                          fc=ORANGE_LT, ec=ORANGE, lw=1.5, zorder=2)
    ax.add_patch(desc)
    ax.text(7.0, 1.2, "v1 used up to 5 Groq calls (3-tier fallback + 2 refinement passes)", fontsize=13, color=DARK, ha="center", va="center")
    ax.text(7.0, 0.6, "v2 uses 1-2 calls: one vision call with built-in self-critique, optional text refinement", fontsize=13, fontweight="bold", color=ORANGE, ha="center", va="center")

    save(fig, "02_mapping_refinement")


# ════════════════════════════════════════════════════════════
# 3. RENDERER CLIP TYPES (v2: Ken Burns + Gap-Freeze)
# ════════════════════════════════════════════════════════════
def diagram_renderer():
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.set_facecolor(WHITE)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7.5, 7.2, "Timeline Rendering: Clip Types (v2)", ha="center",
            fontsize=24, fontweight="bold", color=NAVY)

    # 1. Content Clip
    rounded_box(ax, 0.3, 3.5, 3.5, 2.0, "Content Clip", "Speed: 0.5x - 2.5x",
                fc=SKY, ec=ACCENT, tc=NAVY, fs=19, sub_fs=13)
    ax.text(2.05, 2.8, "Normal playback adjusted\nto match audio length",
            fontsize=11, color=DARK, ha="center")

    # 2. Freeze Clip (with Ken Burns)
    rounded_box(ax, 4.1, 3.5, 3.5, 2.0, "Freeze + Ken Burns", "Subtle zoom 1.0x->1.03x",
                fc=ORANGE_LT, ec=ORANGE, tc=ORANGE, fs=17, sub_fs=12, lw=3)
    ax.text(5.85, 2.8, "Slow zoom on keyframe\nprevents dead-screen effect",
            fontsize=11, color=DARK, ha="center")
    # v2 badge
    badge = FancyBboxPatch((7.0, 5.2), 0.9, 0.4, boxstyle="round,pad=0.06",
                            fc=PURPLE, ec=PURPLE, lw=0, zorder=5)
    ax.add_patch(badge)
    ax.text(7.45, 5.4, "NEW", ha="center", va="center",
            fontsize=9, fontweight="bold", color=WHITE, zorder=6)

    # 3. Gap Clip (normal)
    rounded_box(ax, 7.9, 3.5, 3.5, 2.0, "Gap Clip", "Smooth speed + silence",
                fc=ICE, ec=GRAY, tc=DARK, fs=19, sub_fs=13)
    ax.text(9.65, 2.8, "Video plays normally\nduring narrator pauses",
            fontsize=11, color=DARK, ha="center")

    # 4. Gap-Freeze (NEW)
    rounded_box(ax, 11.7, 3.5, 3.0, 2.0, "Gap-Freeze", "Still frame + silence",
                fc=PURPLE_LT, ec=PURPLE, tc=PURPLE, fs=17, sub_fs=12, lw=2)
    ax.text(13.2, 2.8, "Clean hold when video\ntoo short for smooth play",
            fontsize=11, color=DARK, ha="center")
    badge2 = FancyBboxPatch((14.0, 5.2), 0.9, 0.4, boxstyle="round,pad=0.06",
                            fc=PURPLE, ec=PURPLE, lw=0, zorder=5)
    ax.add_patch(badge2)
    ax.text(14.45, 5.4, "NEW", ha="center", va="center",
            fontsize=9, fontweight="bold", color=WHITE, zorder=6)

    # Bottom: all concat
    for cx in [2.05, 5.85, 9.65, 13.2]:
        ax.annotate("", xy=(7.5, 1.2), xytext=(cx, 3.5),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.5", color=NAVY, lw=2,
                                    connectionstyle="arc3,rad=0"), zorder=1)

    ax.text(7.5, 0.6, "Concatenated into final .mp4 (stream copy, no re-encoding)",
            fontsize=15, fontweight="bold", color=GREEN, ha="center")

    save(fig, "03_renderer_clips")


if __name__ == "__main__":
    print("Generating Showtime Diagrams (v2)...")
    diagram_pipeline()
    diagram_ai_mapper()
    diagram_renderer()
    print("Done!")
