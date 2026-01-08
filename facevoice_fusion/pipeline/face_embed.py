from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from .utils import console


def _load_frame(video_path: Path, timestamp: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_idx = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Unable to read frame for embedding")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def embed_faces(video_path: Path, tracks: List[dict], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    embeddings: Dict[str, Path] = {}

    for track in tracks:
        if not track["bboxes"]:
            continue
        first_box = track["bboxes"][0]
        frame = _load_frame(video_path, first_box["t"])
        x, y, w, h = first_box["x"], first_box["y"], first_box["w"], first_box["h"]
        crop = frame[max(y, 0) : y + h, max(x, 0) : x + w]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (160, 160))
        tensor = torch.tensor(crop_resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            embedding = resnet(tensor).squeeze(0).numpy()
        emb_path = output_dir / f"{track['track_id']}.npy"
        np.save(emb_path, embedding)
        embeddings[track["track_id"]] = emb_path
        console.log("Saved embedding", track=track["track_id"], path=str(emb_path))

    return embeddings
