# face-voice-emotion Fusion

Automatic Person, Emotion & Voice Detection from Video

FaceVoice Fusion is an end-to-end AI system that processes an uploaded raw video and automatically:

Detects and tracks people (faces) over time

Identifies known persons from an identity store (auto-naming)

Detects emotions per person over time

Extracts audio and performs speaker diarization

Associates who is speaking with who is visible

Streams results live to the UI

Exports a UI-ready JSON for playback and review

The system is designed to learn identities over time with human-in-the-loop confirmation.

✨ Key Features

✅ Upload a video → everything runs automatically

✅ Face tracking with persistent IDs (FT1, FT2…)

✅ Identity recognition (name shown if already known)

✅ Emotion detection with transformer-based facial expression model (`trpakov/vit-face-expression`) + confidence overlay

✅ Voice diarization (S1, S2…)

✅ Speaker ↔ face association (talking head mapping)

✅ Real-time progress & events via SSE

✅ Re-identification across videos

✅ Windows-friendly, CPU-first

🏗️ Architecture Overview

Upload Video

   ↓

Face Detect + Track ───────────────┐

   ↓                               │
Face Embeddings → Identity Match 
  │
   ↓                               │
Emotion Detection                  │
                                   ├─→ UI Overlay + Timeline
Audio Extract → Diarization        │
   ↓                               │
Speaker ↔ Face Association ────────┘
   ↓
Export ui.json + persist identity store

📁 Repository Structure
facevoice_fusion/
  app/                # FastAPI backend + APIs
  pipeline/           # Video processing pipeline stages
  identity_store/     # Persistent identity database
  data/               # Uploaded videos & job artifacts
  env/                # requirements.txt
  scripts/            # install/run helpers
  .vscode/            # VS Code config
  README.md

🧠 Models Used
Face Detection & Tracking

- Detector: facenet-pytorch MTCNN for face detection.
- Embedder: facenet-pytorch InceptionResnetV1 (`vggface2`) for face embeddings/identity matching.

Emotion Detection

- Transformer model: `trpakov/vit-face-expression` (Hugging Face). The app accepts `EMOTION_MODEL_NAME=nateraw/fer` but maps it to `trpakov/vit-face-expression` for compatibility.

🧰 Technologies Used (and Why)

| Technology | Why it is used |
|---|---|
| **Python 3.11** | Main backend and pipeline language; strong support for AI/ML and media tooling. |
| **FastAPI + Uvicorn** | Lightweight async API server for upload, job management, and streaming events (SSE). |
| **PyTorch / TorchVision / Torchaudio** | Core deep-learning runtime used by face and emotion models. |
| **facenet-pytorch (MTCNN + InceptionResnetV1)** | MTCNN detects faces; InceptionResnetV1 creates robust face embeddings for identity matching and re-identification. |
| **Transformers (Hugging Face)** | Runs the facial emotion classifier (`trpakov/vit-face-expression`) for per-frame/per-track emotion labels. |
| **OpenCV** | Video frame decode/processing, drawing overlays, and track-level visual operations. |
| **FFmpeg** | Reliable audio extraction and media conversion before diarization/transcription steps. |
| **NumPy (pinned)** | Stable numerical operations and array handling across pipeline stages. |
| **Rich** | Better structured logs and progress output for local development/debugging. |
| **HTML + JavaScript** | Simple browser UI for timeline playback and result visualization. |
| **pyannote.audio (optional)** | Higher-quality speaker diarization when a Hugging Face token is provided. |

---

## 🔄 Workflow (End-to-End)

1. **Video upload**
   - User uploads a raw video through the API/UI.
   - A job is created and queued.

2. **Frame pipeline starts**
   - Frames are read from video.
   - Faces are detected and tracked with stable IDs (FT1, FT2, ...).

3. **Identity inference**
   - Face embeddings are computed for each track.
   - Embeddings are compared with the identity store.
   - Known match → show person name; unknown → keep as Unknown and allow later confirmation.

4. **Emotion inference**
   - Cropped face regions are passed to the emotion model.
   - Emotions + confidence are added to each face track timeline.

5. **Audio pipeline**
   - Audio is extracted using FFmpeg.
   - Speaker diarization segments the audio into speaker IDs (S1, S2, ...).

6. **Speaker-to-face association**
   - Temporal overlap and activity heuristics associate active speakers with visible face tracks.
   - Produces "who is speaking" alignment for playback.

7. **Streaming updates**
   - Pipeline progress and intermediate events are emitted via SSE (`/jobs/{job_id}/events`).

8. **Final export**
   - `ui.json` is generated with tracks, names, emotions, speaker segments, and associations.
   - Identity store is updated for future re-identification across videos.

---

⚙️ Prerequisites
Required

Python 3.10 or 3.11

FFmpeg installed and in PATH

Windows: https://ffmpeg.org/download.html

macOS: brew install ffmpeg

Linux: sudo apt install ffmpeg

Internet access on first run to download the emotion model weights from Hugging Face

Optional

Hugging Face token (for real diarization)

setx HUGGINGFACE_TOKEN hf_xxx   # Windows
export HUGGINGFACE_TOKEN=hf_xxx # macOS/Linux

Emotion model override (defaults to `nateraw/fer`)
export EMOTION_MODEL_NAME=nateraw/fer

🚀 Installation
1) Clone Repository
git clone <your-repo-url>
cd facevoice_fusion

2) Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # macOS/Linux

3) Install Dependencies
pip install --upgrade pip
pip install -r env/requirements.txt

▶️ Running the Server
Option A: VS Code (Recommended)

Open folder in VS Code

Select Python interpreter from .venv

Press F5 (launch config included)

Option B: Terminal
uvicorn app.main:app --reload


Server runs at:

http://localhost:8000

📤 Upload a Video (Test)
Using curl
curl -X POST http://localhost:8000/upload \
  -F "file=@sample.mp4"


Response:

{
  "job_id": "abc123",
  "video_id": "vid_001",
  "status": "queued"
}

📡 Live Progress & Events

Subscribe to Server-Sent Events:

GET /jobs/{job_id}/events


Events include:

job.progress

track.created

track.identity

track.emotion

speaker.segment

association.updated

job.done

📄 UI Output (Final Result)

When processing completes:

GET /jobs/{job_id}/ui


Returns ui.json with:

Face tracks + bounding boxes

Names (or Unknown)

Emotion timelines

Speaker segments

Speaker ↔ face associations

Artifact paths

Model versions

🧑 Identity Store (Auto-Naming)

Stored at:

identity_store/identities.json

Enroll a New Person
POST /identity/enroll
{
  "job_id": "abc123",
  "track_id": "FT3",
  "name": "Micke"
}


Once enrolled:

Future videos auto-label that person

🧪 Fallback Behavior (By Design)

No HuggingFace token → diarization falls back to single speaker

Emotion model downloads from Hugging Face on first run (requires network access)

No GPU → CPU inference only

This guarantees the system always runs.

🛠️ Troubleshooting
NumPy errors (_ARRAY_API not found)

✔ Ensure:

numpy==1.26.4

FFmpeg not found

✔ Confirm:

ffmpeg -version

Torch issues on Windows

✔ Use CPU-only torch builds
✔ Python 3.11 recommended

🗺️ Roadmap

Real-time webcam mode

Faster emotion model

GPU acceleration

React overlay player

Privacy controls (face blur, redaction)

Multi-language speech support

📜 License

MIT (or your choice)

🙌 Credits

Built with modern open-source ML tools and a human-in-the-loop design philosophy.
