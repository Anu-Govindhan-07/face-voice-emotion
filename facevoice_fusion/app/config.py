from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
IDENTITY_STORE = BASE_DIR / "identity_store" / "identities.json"
STATIC_DIR = BASE_DIR / "static"

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

FACE_MATCH_THRESHOLD = 0.55
FACE_MAYBE_THRESHOLD = 0.45

AUDIO_SAMPLE_RATE = 16000

MODEL_VERSIONS = {
    "face_detector": "facenet-pytorch-mtcnn",
    "face_embedder": "facenet-pytorch-inceptionresnetv1",
    "emotion_model": "stub-neutral-v1",
    "diarization_model": "pyannote-or-fallback",
}
