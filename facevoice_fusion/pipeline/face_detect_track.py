from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from facenet_pytorch import MTCNN

from app.config import (
    FACE_DETECT_MAX_DIM,
    FACE_DETECT_MIN_CONF,
    FACE_DETECT_MIN_SIZE,
    FACE_DETECT_SAMPLE_EVERY,
)
from .utils import console, save_json


@dataclass
class Track:
    track_id: str
    bboxes: List[dict] = field(default_factory=list)
    last_box: Tuple[int, int, int, int] | None = None
    last_t: float = 0.0


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter_area / float(area_a + area_b - inter_area)


def _nms(
    detections: List[Tuple[Tuple[int, int, int, int], float]],
    threshold: float,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda item: item[1], reverse=True)
    keep: List[Tuple[Tuple[int, int, int, int], float]] = []
    while ordered:
        current = ordered.pop(0)
        keep.append(current)
        remaining = []
        for candidate in ordered:
            if _iou(current[0], candidate[0]) <= threshold:
                remaining.append(candidate)
        ordered = remaining
    return keep


def detect_and_track(video_path: Path, output_path: Path) -> Path:
    console.log(f"Running face detection and tracking for {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    mtcnn = MTCNN(keep_all=True, device="cpu")

    tracks: List[Track] = []
    frame_idx = 0
    sample_every = max(1, FACE_DETECT_SAMPLE_EVERY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue
        scale = 1.0
        if FACE_DETECT_MAX_DIM > 0:
            height, width = frame.shape[:2]
            max_dim = max(height, width)
            if max_dim > FACE_DETECT_MAX_DIM:
                scale = FACE_DETECT_MAX_DIM / float(max_dim)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb)
        timestamp = frame_idx / fps
        detections: List[Tuple[Tuple[int, int, int, int], float]] = []
        if boxes is not None:
            for box, conf in zip(boxes, probs):
                if conf is None:
                    continue
                conf_value = float(conf)
                if conf_value < FACE_DETECT_MIN_CONF:
                    continue
                if scale != 1.0:
                    box = box / scale
                x1, y1, x2, y2 = [int(v) for v in box]
                width = x2 - x1
                height = y2 - y1
                if width < FACE_DETECT_MIN_SIZE or height < FACE_DETECT_MIN_SIZE:
                    continue
                detections.append(((x1, y1, width, height), conf_value))
        detections = _nms(detections, 0.4)

        unmatched = detections.copy()
        for track in tracks:
            best_iou = 0.0
            best_det = None
            for det in unmatched:
                iou = _iou(track.last_box or det[0], det[0])
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            if best_det and best_iou > 0.3:
                box, conf = best_det
                track.bboxes.append({
                    "t": timestamp,
                    "x": box[0],
                    "y": box[1],
                    "w": box[2],
                    "h": box[3],
                    "conf": conf,
                })
                track.last_box = box
                track.last_t = timestamp
                unmatched.remove(best_det)

        for det in unmatched:
            box, conf = det
            track_id = f"FT{len(tracks) + 1}"
            track = Track(track_id=track_id, last_box=box, last_t=timestamp)
            track.bboxes.append({
                "t": timestamp,
                "x": box[0],
                "y": box[1],
                "w": box[2],
                "h": box[3],
                "conf": conf,
            })
            tracks.append(track)

        frame_idx += 1

    cap.release()

    payload = {
        "tracks": [
            {
                "track_id": track.track_id,
                "start": track.bboxes[0]["t"] if track.bboxes else 0.0,
                "end": track.bboxes[-1]["t"] if track.bboxes else 0.0,
                "bboxes": track.bboxes,
            }
            for track in tracks
        ]
    }
    save_json(output_path, payload)
    return output_path
