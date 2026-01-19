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

✅ Emotion detection with transformer-based facial expression model + confidence overlay

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

🧰 Tech Stack
Languages

Python 3.11 (backend + ML pipeline)

JavaScript / HTML (minimal UI viewer)

Core Libraries

FastAPI, Uvicorn

NumPy ==1.26.4 (pinned)

Torch / TorchVision / Torchaudio (CPU)

OpenCV

Transformers (Hugging Face) for facial expression recognition (`nateraw/fer`)

facenet-pytorch (MTCNN + FaceNet embeddings)

FFmpeg

Rich (logging)

Optional

pyannote.audio (speaker diarization, requires HF token)

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
