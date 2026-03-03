from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from .utils import console


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _bbox_area(bb: dict) -> float:
    return float(bb.get("w", 0.0)) * float(bb.get("h", 0.0))


def _read_frame(cap: cv2.VideoCapture, fps: float, ts: float) -> Optional[np.ndarray]:
    frame_idx = int(max(0.0, ts) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _crop_face(rgb: np.ndarray, bb: dict, pad: float = 0.12) -> Optional[np.ndarray]:
    h, w = rgb.shape[:2]

    x = _safe_int(bb.get("x", 0))
    y = _safe_int(bb.get("y", 0))
    bw = _safe_int(bb.get("w", 0))
    bh = _safe_int(bb.get("h", 0))

    if bw <= 1 or bh <= 1:
        return None

    px = int(bw * pad)
    py = int(bh * pad)

    x0 = _clamp(x - px, 0, max(0, w - 1))
    y0 = _clamp(y - py, 0, max(0, h - 1))
    x1 = _clamp(x + bw + px, 0, w)
    y1 = _clamp(y + bh + py, 0, h)

    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def _choose_sample_bboxes(
    bboxes: List[dict],
    max_samples: int = 8,
    min_area: float = 60 * 60,
) -> List[dict]:
    if not bboxes:
        return []

    filtered = [bb for bb in bboxes if _bbox_area(bb) >= float(min_area)]
    if not filtered:
        filtered = list(bboxes)

    filtered = sorted(filtered, key=lambda b: float(b.get("t", 0.0)))

    if len(filtered) <= max_samples:
        return filtered

    idxs = np.linspace(0, len(filtered) - 1, num=max_samples).round().astype(int).tolist()
    samples = [filtered[i] for i in idxs]

    biggest = sorted(filtered, key=_bbox_area, reverse=True)[:2]
    for bb in biggest:
        if bb not in samples:
            samples.append(bb)

    samples = sorted(samples, key=lambda b: float(b.get("t", 0.0)))
    return samples[:max_samples]


def _embed_crops(model: InceptionResnetV1, crops: List[np.ndarray], device: torch.device) -> Optional[np.ndarray]:
    if not crops:
        return None

    tensors = []
    for crop in crops:
        crop_resized = cv2.resize(crop, (160, 160))
        t = torch.tensor(crop_resized).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad():
        emb = model(batch).detach().cpu().numpy()

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    mean = emb.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)
    return mean.astype(np.float32)


def embed_track_window(
    video_path: Path,
    track: dict,
    start_ts: float,
    end_ts: float,
    output_path: Optional[Path] = None,
    max_samples: int = 6,
) -> Optional[Path]:
    """
    Create an embedding for one track within a specific time window.
    Used for clean identity enrollment from self-intro segments.
    """
    bboxes = [
        bb for bb in (track.get("bboxes") or [])
        if float(start_ts) <= float(bb.get("t", -1.0)) <= float(end_ts)
    ]
    if not bboxes:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    sample_bbs = _choose_sample_bboxes(bboxes, max_samples=max_samples)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    crops: List[np.ndarray] = []
    try:
        for bb in sample_bbs:
            ts = float(bb.get("t", 0.0))
            frame = _read_frame(cap, fps, ts)
            if frame is None:
                continue
            crop = _crop_face(frame, bb, pad=0.12)
            if crop is None:
                continue
            crops.append(crop)
    finally:
        cap.release()

    emb = _embed_crops(model, crops, device)
    if emb is None:
        return None

    if output_path is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), emb)
    return output_path


def embed_faces(video_path: Path, tracks: List[dict], output_dir: Path) -> Dict[str, Path]:
    """
    Improved track embedding:
    - multiple frames per track
    - average normalized embeddings
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    embeddings: Dict[str, Path] = {}

    try:
        for track in tracks:
            track_id = str(track.get("track_id") or "")
            bboxes = track.get("bboxes") or []
            if not track_id or not bboxes:
                continue

            sample_bbs = _choose_sample_bboxes(bboxes, max_samples=8)

            crops: List[np.ndarray] = []
            for bb in sample_bbs:
                ts = float(bb.get("t", 0.0))
                frame = _read_frame(cap, fps, ts)
                if frame is None:
                    continue
                crop = _crop_face(frame, bb, pad=0.12)
                if crop is None:
                    continue
                crops.append(crop)

            emb = _embed_crops(model, crops, device)
            if emb is None:
                continue

            emb_path = output_dir / f"{track_id}.npy"
            np.save(str(emb_path), emb)
            embeddings[track_id] = emb_path
            console.log(f"Saved embedding for track {track_id} at {emb_path} using {len(crops)} samples")

    finally:
        cap.release()

    return embeddings