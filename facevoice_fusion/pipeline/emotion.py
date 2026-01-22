from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from app.config import HUGGINGFACE_TOKEN
from .utils import console, save_json


EMOTION_MODEL_NAME = os.getenv("EMOTION_MODEL_NAME", "nateraw/fer")
EMOTION_LABELS = {"happy", "sad", "anger", "fear", "disgust", "surprise", "neutral"}
LABEL_NORMALIZATION = {
    "angry": "anger",
    "surprised": "surprise",
}
SAMPLE_EVERY = 3

_emotion_classifier = None


def _get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        console.log(f"Loading emotion model: {EMOTION_MODEL_NAME}")
        pipeline_kwargs = {"device": -1}
        if HUGGINGFACE_TOKEN:
            pipeline_kwargs["token"] = HUGGINGFACE_TOKEN
        _emotion_classifier = pipeline(
            "image-classification",
            model=EMOTION_MODEL_NAME,
            **pipeline_kwargs,
        )
    return _emotion_classifier


def _normalize_label(label: str) -> str:
    label = label.strip().lower()
    label = LABEL_NORMALIZATION.get(label, label)
    return label if label in EMOTION_LABELS else "neutral"


def _crop_face(frame: np.ndarray, bbox: dict) -> Image.Image | None:
    height, width = frame.shape[:2]
    x1 = max(0, int(bbox.get("x", 0)))
    y1 = max(0, int(bbox.get("y", 0)))
    w = max(0, int(bbox.get("w", 0)))
    h = max(0, int(bbox.get("h", 0)))
    x2 = min(width, x1 + w)
    y2 = min(height, y1 + h)
    if x2 <= x1 or y2 <= y1:
        return None
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _classify_emotion(face: Image.Image) -> Tuple[str, float]:
    classifier = _get_emotion_classifier()
    results = classifier(face, top_k=3)
    if not results:
        return "neutral", 0.0
    best = max(results, key=lambda item: item.get("score", 0.0))
    label = _normalize_label(best.get("label", "neutral"))
    score = float(best.get("score", 0.0))
    return label, score


def _frame_at(cap: cv2.VideoCapture, timestamp: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _dominant_from_timeline(timeline: List[dict]) -> str:
    if not timeline:
        return "neutral"
    scores: Dict[str, List[float]] = {}
    for entry in timeline:
        scores.setdefault(entry["label"], []).append(entry.get("conf", 0.0))
    averaged = {label: sum(vals) / max(1, len(vals)) for label, vals in scores.items()}
    return max(averaged.items(), key=lambda item: item[1])[0]


def infer_emotions(tracks: List[dict], output_path: Path, video_path: Path) -> Dict[str, dict]:
    console.log("Running emotion inference (transformers)")
    emotions: Dict[str, dict] = {}
    cap = cv2.VideoCapture(str(video_path))

    for track in tracks:
        bboxes = track.get("bboxes", [])
        timeline: List[dict] = []
        sampled = bboxes[::SAMPLE_EVERY] if SAMPLE_EVERY > 1 else bboxes
        for idx, bbox in enumerate(sampled):
            timestamp = float(bbox.get("t", 0.0))
            frame = _frame_at(cap, timestamp)
            if frame is None:
                continue
            face = _crop_face(frame, bbox)
            if face is None:
                continue
            label, conf = _classify_emotion(face)
            next_index = idx + 1
            if next_index < len(sampled):
                end_time = float(sampled[next_index].get("t", timestamp))
            else:
                end_time = float(track.get("end", timestamp))
            timeline.append({
                "start": timestamp,
                "end": end_time,
                "label": label,
                "conf": conf,
            })

        dominant = _dominant_from_timeline(timeline)
        emotions[track["track_id"]] = {
            "dominant": dominant,
            "timeline": timeline,
        }

    cap.release()
    save_json(output_path, {"tracks": emotions})
    return emotions
