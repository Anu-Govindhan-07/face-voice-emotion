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
if HUGGINGFACE_TOKEN and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN

FACE_MATCH_THRESHOLD = 0.55
FACE_MAYBE_THRESHOLD = 0.45

AUDIO_SAMPLE_RATE = 16000
FACE_DETECT_SAMPLE_EVERY = int(os.getenv("FACE_DETECT_SAMPLE_EVERY", "5"))
FACE_DETECT_MAX_DIM = int(os.getenv("FACE_DETECT_MAX_DIM", "720"))
FACE_DETECT_MIN_CONF = float(os.getenv("FACE_DETECT_MIN_CONF", "0.9"))
FACE_DETECT_MIN_SIZE = int(os.getenv("FACE_DETECT_MIN_SIZE", "40"))

_raw_emotion_model = os.getenv("EMOTION_MODEL_NAME", "trpakov/vit-face-expression")
EMOTION_MODEL_NAME = "trpakov/vit-face-expression" if _raw_emotion_model == "nateraw/fer" else _raw_emotion_model
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "openai/whisper-tiny")

MODEL_VERSIONS = {
    "face_detector": "facenet-pytorch-mtcnn",
    "face_embedder": "facenet-pytorch-inceptionresnetv1",
    "emotion_model": EMOTION_MODEL_NAME,
    "diarization_model": "pyannote-or-fallback",
    "asr_model": ASR_MODEL_NAME,
}
