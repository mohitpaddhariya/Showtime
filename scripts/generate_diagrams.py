#!/usr/bin/env python3
"""Generate clean, projector-friendly Showtime Pipeline diagrams.

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
    # Drop shadow
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
# 1. ARCHITECTURE — Overview Flow
# ════════════════════════════════════════════════════════════
def diagram_pipeline():
    fig, ax = plt.subplots(figsize=(12, 11))
    fig.set_facecolor(WHITE)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Title
    ax.text(6, 10.3, "Showtime Pipeline Architecture", ha="center",
            fontsize=26, fontweight="bold", color=NAVY)

    # Inputs
    rounded_box(ax, 1.0, 8.5, 4.0, 1.2, "Raw Screen Recording", "Video (.mp4)",
                fc=ICE, ec=GRAY, tc=DARK, fs=18)
    rounded_box(ax, 7.0, 8.5, 4.0, 1.2, "Voiceover Audio", "Audio (.mp3)",
                fc=ICE, ec=GRAY, tc=DARK, fs=18)

    arrow_v(ax, 3.0, 8.5, 7.6)
    arrow_v(ax, 9.0, 8.5, 7.6)

    # Step 1 & 3
    rounded_box(ax, 1.0, 6.2, 4.0, 1.4, "1. Scene Detection", "OpenCV: Diff & Split",
                fc=SKY, ec=ACCENT, tc=NAVY, fs=18)
    rounded_box(ax, 7.0, 6.2, 4.0, 1.4, "3. Transcription", "Groq: Whisper (Word-Level)",
                fc=SKY, ec=ACCENT, tc=NAVY, fs=18)

    arrow_v(ax, 3.0, 6.2, 5.3)
    arrow_v(ax, 9.0, 6.2, 5.3)

    # Step 2
    rounded_box(ax, 1.0, 3.9, 4.0, 1.4, "2. Frame Captioning", "Tesseract: OCR & Structure",
                fc=SKY, ec=ACCENT, tc=NAVY, fs=18)

    # Connect Step 2 and 3 into AI Mapper
    # From Step 2 (Captioning, bottom center x=3.0, y=3.9) -> goes down and right to x=6.0
    ax.plot([3.0, 3.0], [3.9, 3.3], color=NAVY, lw=2.5, zorder=1)
    ax.plot([3.0, 6.0], [3.3, 3.3], color=NAVY, lw=2.5, zorder=1)

    # From Step 3 (Transcription, bottom center x=9.0, y=6.2) -> goes down and left to x=6.0
    ax.plot([9.0, 9.0], [6.2, 3.3], color=NAVY, lw=2.5, zorder=1)
    ax.plot([9.0, 6.0], [3.3, 3.3], color=NAVY, lw=2.5, zorder=1)

    # Plunge into Step 4 (AI Mapper, top center x=6.0, y=2.7)
    arrow_v(ax, 6.0, 3.3, 2.7, color=NAVY, lw=2.5)

    # Step 4
    rounded_box(ax, 3.0, 1.2, 6.0, 1.5, "4. AI Mapping & Refinement",
                "Llama 4 Scout (Vision) + Pacing Rules",
                fc=ORANGE_LT, ec=ORANGE, tc=ORANGE, fs=20, sub_fs=13, lw=3)
    
    # Badge
    badge = FancyBboxPatch((8.3, 2.3), 1.5, 0.5, boxstyle="round,pad=0.08",
                            fc=ORANGE, ec=ORANGE, lw=0, zorder=5)
    ax.add_patch(badge)
    ax.text(9.05, 2.55, "CORE AI", ha="center", va="center",
            fontsize=11, fontweight="bold", color=WHITE, zorder=6)

    # Down from Mapper
    arrow_v(ax, 6.0, 1.2, 0.4)

    ax.text(6.0, 0.0, "Timeline Assembly & Rendering", ha="center",
            fontsize=18, fontweight="bold", color=GREEN)

    save(fig, "01_pipeline_overview")


# ════════════════════════════════════════════════════════════
# 2. AI MAPPER REFINEMENT LOOP
# ════════════════════════════════════════════════════════════
def diagram_ai_mapper():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.set_facecolor(WHITE)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7, 7.3, "AI Mapping Refinement Loop", ha="center", va="center",
            fontsize=24, fontweight="bold", color=NAVY)

    # Step A: Initial
    rounded_box(ax, 0.5, 3.5, 3.0, 1.5, "Initial Mapping", "Llama matches\naudio to visual",
                fc=SKY, ec=NAVY, tc=NAVY, fs=18, sub_fs=12)

    arrow_h(ax, 3.5, 4.25, 5.5, NAVY)
    ax.text(4.5, 4.5, "Sends EDL", ha="center", va="center", fontsize=11, fontweight="bold", color=DARK)

    # Step B: Pacing Analysis
    rounded_box(ax, 5.5, 3.5, 3.0, 1.5, "Pacing Analysis", "Checks speed limits\n& min durations",
                fc=ICE, ec=GRAY, tc=DARK, fs=18, sub_fs=12)

    # Feedback Loop (Reject) - orthogonal flow
    ax.plot([7.0, 7.0], [5.0, 5.8], color=RED, lw=2.5, zorder=1)
    ax.plot([7.0, 2.0], [5.8, 5.8], color=RED, lw=2.5, zorder=1)
    arrow_v(ax, 2.0, 5.8, 5.0, color=RED, lw=2.5)
    ax.text(4.5, 6.0, "Rejected (Speed > 2.5x)", ha="center", va="bottom", fontsize=13, fontweight="bold", color=RED)

    # Approve (Accept)
    arrow_h(ax, 8.5, 4.25, 10.5, GREEN)
    ax.text(9.5, 4.5, "Approved", ha="center", va="center", fontsize=13, fontweight="bold", color=GREEN)

    # Result
    rounded_box(ax, 10.5, 3.5, 3.0, 1.5, "Final Timeline", "Chronological EDL\nready for FFmpeg",
                fc=GREEN_LT, ec=GREEN, tc=GREEN, fs=18, sub_fs=12)

    # Explanation text
    desc = FancyBboxPatch((0.5, 0.2), 13.0, 2.4, boxstyle="round,pad=0.15",
                          fc=ORANGE_LT, ec=ORANGE, lw=1.5, zorder=2)
    ax.add_patch(desc)
    ax.text(7.0, 2.2, "Why this matters?", fontsize=16, fontweight="bold", color=ORANGE, ha="center", va="center")
    
    body_text = (
        "Without the refinement loop, an AI might map 3 long sentences into a tiny 4-second \n"
        "video clip, which would force the final video to play at an excessive 300% speed.\n\n"
        "The Pacing Analysis catches these extreme speeds, rejects the timeline, and forces\n"
        "the AI Mapper to choose a longer visual segment to guarantee smooth playback."
    )
    # y=1.2 gives it a perfect center under the title
    ax.text(7.0, 1.2, body_text, fontsize=13, color=DARK, ha="center", va="center", linespacing=1.6)

    save(fig, "02_mapping_refinement")


# ════════════════════════════════════════════════════════════
# 3. RENDERER TRICKS (Auto-freeze, Gaps)
# ════════════════════════════════════════════════════════════
def diagram_renderer():
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor(WHITE)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(7, 6.2, "Timeline Rendering: Clip Types", ha="center",
            fontsize=24, fontweight="bold", color=NAVY)

    # 1. Content Clip
    rounded_box(ax, 0.5, 2.5, 4.0, 2.0, "Content Clip", "Speed: 0.5x - 2.5x",
                fc=SKY, ec=ACCENT, tc=NAVY, fs=20, sub_fs=13)
    ax.text(2.5, 1.8, "\u2022 Normal playback adjusted\n   to perfectly match audio length",
            fontsize=12, color=DARK, ha="center")

    # 2. Gap Clip
    rounded_box(ax, 5.0, 2.5, 4.0, 2.0, "Gap Clip", "Silent Pause > 0.15s",
                fc=ICE, ec=GRAY, tc=DARK, fs=20, sub_fs=13)
    ax.text(7.0, 1.8, "\u2022 Video continues playing normally\n   while audio goes silent",
            fontsize=12, color=DARK, ha="center")

    # 3. Auto-Freeze
    rounded_box(ax, 9.5, 2.5, 4.0, 2.0, "Freeze Clip", "Speed > 2.5x OR chosen by AI",
                fc=ORANGE_LT, ec=ORANGE, tc=ORANGE, fs=20, sub_fs=13, lw=3)
    ax.text(11.5, 1.8, "\u2022 Holds exactly one keyframe still\n   so video doesn't become a blur",
            fontsize=12, color=DARK, ha="center")

    # Bottom concat (using connection styles for clean flowchart routing)
    ax.annotate("", xy=(7.0, 0.3), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7", color=NAVY, lw=2.5,
                                connectionstyle="angle,angleA=270,angleB=180,rad=10"), zorder=1)
    arrow_v(ax, 7.0, 2.5, 0.3, NAVY, 2.5)
    ax.annotate("", xy=(7.0, 0.3), xytext=(11.5, 2.5),
                arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7", color=NAVY, lw=2.5,
                                connectionstyle="angle,angleA=270,angleB=0,rad=10"), zorder=1)

    ax.text(7.0, -0.2, "Concatenated into final .mp4 (No re-encoding audio)",
            fontsize=14, fontweight="bold", color=GREEN, ha="center")

    save(fig, "03_renderer_clips")


if __name__ == "__main__":
    print("Generating Showtime Diagrams...")
    diagram_pipeline()
    diagram_ai_mapper()
    diagram_renderer()
    print("Done!")
