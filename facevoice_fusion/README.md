# FaceVoice Fusion

End-to-end pipeline for uploading a video, detecting/tracking faces, matching identities, estimating emotions, diarizing speakers, associating speakers to faces, streaming results via SSE, and exporting a UI-ready JSON file.

## Prerequisites
- **Python 3.11** installed and available on PATH
- **FFmpeg** installed and available on PATH
  - Windows: https://ffmpeg.org/download.html
  - macOS: `brew install ffmpeg`
  - Linux: use your distro package manager
- **Internet access on first run** to download the emotion model weights from Hugging Face

## Setup

### Windows PowerShell
```powershell
cd facevoice_fusion
./scripts/install.ps1
```

### macOS/Linux
```bash
cd facevoice_fusion
./scripts/install.sh
```

## Run the server
```bash
cd facevoice_fusion
./scripts/run_server.sh
```

Or on Windows:
```powershell
cd facevoice_fusion
./scripts/run_server.ps1
```

The API will be available at `http://localhost:8000`.

## Emotion model configuration
The pipeline uses the Hugging Face model `nateraw/fer` by default. Override with:

```bash
export EMOTION_MODEL_NAME=nateraw/fer
```

## Face detection tuning
If you see false face boxes, raise the minimum confidence or size thresholds:

```bash
export FACE_DETECT_MIN_CONF=0.9
export FACE_DETECT_MIN_SIZE=40
```

## Upload a video (curl)
```bash
curl -F "file=@/path/to/video.mp4" http://localhost:8000/upload
```

Then check job status:
```bash
curl http://localhost:8000/jobs/<job_id>
```

When done, download UI JSON:
```bash
curl http://localhost:8000/jobs/<job_id>/ui
```

## Outputs
All artifacts are stored under:
```
facevoice_fusion/data/jobs/<job_id>/
```
Including:
- `audio.wav`
- `faces_tracks.json`
- `emotions.json`
- `diarization.json`
- `associations.json`
- `ui.json`

## Identity store
The identity store lives in `identity_store/identities.json` and acts as a persistent memory layer:
- Every processed face embedding is stored automatically (even before a name is assigned).
- If a later upload matches an existing face and that identity has a name, the matched name is rendered in the face-tracking overlay label.
- The pipeline runs ASR and infers spoken names for association metadata only by default; it does **not** auto-rename/enroll unknown faces from speech unless explicitly enabled.
- To enable speech-driven auto-enrollment (higher risk of wrong labels), set `ALLOW_TRANSCRIPT_IDENTITY_ENROLL=1`.
- Matching is optimized with per-person centroid indexes and an in-memory embedding cache for better throughput on large/high-quality videos.

You can assign or update a name for a detected track:
```bash
curl -X POST http://localhost:8000/identity/enroll \
  -H "Content-Type: application/json" \
  -d '{"job_id": "<job_id>", "track_id": "FT1", "name": "Micke"}'
```

List identities:
```bash
curl http://localhost:8000/identity/list
```

## Optional: Hugging Face diarization
If you have a token, set it before running:

### Windows PowerShell
```powershell
$env:HUGGINGFACE_TOKEN="your_token"
```

### macOS/Linux
```bash
export HUGGINGFACE_TOKEN="your_token"
```

If the token is missing or pyannote is unavailable, the pipeline falls back to a single-speaker diarization.

## Optional: Emotion model selection
The default emotion model is `trpakov/vit-face-expression`. You can override it by setting:

```bash
export EMOTION_MODEL_NAME="your-hf-model-id"
```

## Troubleshooting
- **NumPy pin**: The project pins `numpy==1.26.4` to avoid breaking changes in 2.x.
- **FFmpeg not found**: Ensure `ffmpeg` is installed and on PATH.
- **Torch CPU builds**: The requirements use CPU-friendly builds; GPU is not required.

## UI Viewer
Open `http://localhost:8000` for a minimal HTML viewer that uploads a video, streams progress via SSE, and links to `ui.json`.
