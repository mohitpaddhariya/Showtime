# Showtime — Implementation Plan

## Overview

Showtime turns a rough screen recording into a polished, synced demo video — automatically.

---

## Phases

### Phase 1 — MVP (Current)

User uploads **screen recording** + **voiceover audio** → gets a polished demo video.

### Phase 2 — Future

User uploads **only a raw screen recording** → Showtime helps generate a transcript → generates voiceover audio → produces the final video.

---

## Phase 1 — MVP

### User Flow

```
1. User uploads screen recording (.mp4/.mov)
2. User uploads voiceover audio (.mp3/.wav)
3. Showtime processes both
4. User downloads the final polished video
```

### Processing Pipeline

```
Screen Recording                    Voiceover Audio
      |                                   |
      v                                   v
Scene Detection (OpenCV)         Transcription (Whisper)
      |                                   |
      v                                   v
Keyframe Extraction              Sentence Segmentation
      |                            (with timestamps)
      v                                   |
OCR / Frame Description                   |
  (Tesseract)                             |
      |                                   |
      v                                   v
      +----------→ AI Mapper ←------------+
                  (Ollama/Llama 3)
                      |
                      v
              Timeline Assembly
                      |
                      v
                FFmpeg Render
                      |
                      v
                Final Video (.mp4)
```

### Step-by-Step Breakdown

#### Step 1 — Screen Recording Analysis

- **Input:** Raw screen recording video
- **Process:**
  - Use OpenCV to compute frame-by-frame pixel difference
  - Detect scene changes where difference exceeds a threshold
  - Identify idle/dead segments (low pixel delta over consecutive frames)
  - Extract one keyframe per scene
  - Run Tesseract OCR on each keyframe to get text description of what's on screen
- **Output:**
  ```json
  [
    {
      "segment_id": 1,
      "start": 0.0,
      "end": 5.2,
      "description": "Landing page with sign-up button visible",
      "is_idle": false,
      "keyframe_path": "frames/segment_001.png"
    }
  ]
  ```

#### Step 2 — Voiceover Analysis

- **Input:** Voiceover audio file
- **Process:**
  - Run Whisper (local model) to transcribe with word-level timestamps
  - Group words into sentence-level segments
- **Output:**
  ```json
  [
    {
      "sentence_id": 1,
      "text": "Here's our landing page where users can sign up.",
      "start": 0.0,
      "end": 3.1
    }
  ]
  ```

#### Step 3 — AI Mapping

- **Input:** Screen segments + voiceover sentences
- **Process:**
  - Send both lists to a local LLM (Ollama + Llama 3)
  - Prompt the model to match each voiceover sentence to the best screen segment(s)
  - Model returns a JSON mapping with optional speed adjustments
- **Prompt structure:**
  ```
  You are a video editor. Given these screen segments and voiceover
  sentences, map each sentence to the screen segment that best matches
  what the narrator is describing.

  Screen segments: [...]
  Voiceover sentences: [...]

  Return a JSON array of: { sentence_id, segment_id, speed_factor }
  ```
- **Output:**
  ```json
  [
    {
      "sentence_id": 1,
      "segment_id": 1,
      "speed_factor": 1.0
    },
    {
      "sentence_id": 2,
      "segment_id": 3,
      "speed_factor": 1.5
    }
  ]
  ```

#### Step 4 — Timeline Assembly & Render

- **Input:** AI mapping + original video + audio
- **Process:**
  - Build edit decision list from mapping
  - **Timeline:** Allocate video slices to sentences (enforcing a minimum video duration per clip)
  - **Renderer:** For each mapping entry:
    - Extract the screen segment and adjust playback speed to match voiceover duration
    - If speed exceeds max threshold (2.5x), **auto-freeze** on a keyframe to prevent choppy unwatchable playback
  - Concatenate all adjusted clips sequentially
  - Overlay the voiceover audio track
  - Render to MP4 (H.264 + AAC)
- **Output:** Final polished video file

---

## Phase 2 — Smart Transcript + Audio Generation

### User Flow

```
1. User uploads raw screen recording
2. Showtime analyzes the recording
3. Showtime generates a draft transcript/script describing the demo
4. User reviews and edits the transcript
5. User generates voiceover from the transcript (via ElevenLabs or local TTS)
6. Showtime syncs and renders the final video
```

### Additional Pipeline Steps

#### Step A — Auto Script Generation

- **Input:** Screen segments with OCR descriptions (from Phase 1 Step 1)
- **Process:**
  - Send all segment descriptions to the LLM
  - Prompt: "Write a short, professional voiceover script for this screen recording demo"
  - LLM generates a sentence per key screen moment
- **Output:** Draft script the user can edit in the UI

#### Step B — Voiceover Generation

- **Option 1: ElevenLabs (user's API key)**
  - User pastes their ElevenLabs API key
  - Showtime sends the final script to ElevenLabs
  - Returns generated audio
- **Option 2: Local TTS (free)**
  - Use Coqui TTS or Piper (open source, runs locally)
  - Lower quality but completely free
- **Option 3: Manual upload**
  - User generates audio themselves and uploads it (same as Phase 1)

---

## Tech Stack

| Component          | Tool                  | Cost  |
|--------------------|-----------------------|-------|
| Backend            | Python + FastAPI      | Free  |
| Frontend           | Next.js               | Free  |
| Scene Detection    | OpenCV                | Free  |
| OCR                | Tesseract             | Free  |
| Transcription      | Whisper (local)       | Free  |
| AI Mapping/Scripts | Ollama + Llama 3      | Free  |
| Video Rendering    | FFmpeg                | Free  |
| Job Queue          | Celery + Redis        | Free  |
| Storage            | Local filesystem      | Free  |
| TTS (Phase 2)     | Coqui TTS / Piper     | Free  |

**Total cost: $0 (fully local)**

---

## Project Structure

```
Showtime/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   ├── upload.py            # File upload endpoints
│   │   ├── process.py           # Trigger processing
│   │   └── download.py          # Serve final video
│   ├── pipeline/
│   │   ├── scene_detector.py    # OpenCV scene detection
│   │   ├── frame_captioner.py   # Tesseract OCR on keyframes
│   │   ├── audio_analyzer.py    # Whisper transcription
│   │   ├── ai_mapper.py         # LLM mapping logic
│   │   ├── timeline.py          # Timeline assembly
│   │   └── renderer.py          # FFmpeg rendering
│   ├── workers/
│   │   └── tasks.py             # Celery async tasks
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Upload page
│   │   ├── processing/
│   │   │   └── page.tsx         # Processing status
│   │   └── result/
│   │       └── page.tsx         # Preview + download
│   └── package.json
├── cli.py                       # CLI tool (for testing)
├── pitch.md
└── implementation.md
```

---

## Build Order

### Week 1 — Core Pipeline (CLI only)

1. `scene_detector.py` — OpenCV scene detection + idle removal
2. `frame_captioner.py` — Tesseract OCR on keyframes
3. `audio_analyzer.py` — Whisper transcription + sentence segmentation
4. `ai_mapper.py` — Ollama/Llama 3 mapping prompt
5. `timeline.py` — Build edit decision list from mapping
6. `renderer.py` — FFmpeg render
7. `cli.py` — Wire it all together

**Milestone:** `python cli.py --video demo.mp4 --audio narration.mp3 --output final.mp4`

### Week 2 — Web App

1. FastAPI backend with upload/process/download endpoints
2. Celery worker for async processing
3. Next.js frontend — upload, progress, preview, download

### Week 3 — Phase 2 Features

1. Auto script generation from screen segments
2. Script editor in the frontend
3. Local TTS integration (Coqui/Piper)
4. Optional ElevenLabs integration

---

## Key Decisions

- **Ollama over cloud APIs** — keeps everything free and local, swap to Claude API later for better accuracy if needed
- **CLI first, web second** — validate the pipeline works before building UI
- **FFmpeg for rendering** — battle-tested, handles all codec/format edge cases
- **Whisper local model** — no API dependency, good accuracy for English voiceovers
