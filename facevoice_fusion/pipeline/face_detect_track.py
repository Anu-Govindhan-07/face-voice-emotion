from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from facenet_pytorch import MTCNN

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


def detect_and_track(video_path: Path, output_path: Path) -> Path:
    console.log("Running face detection and tracking", video=str(video_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    mtcnn = MTCNN(keep_all=True, device="cpu")

    tracks: List[Track] = []
    frame_idx = 0
    sample_every = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb)
        timestamp = frame_idx / fps
        detections: List[Tuple[Tuple[int, int, int, int], float]] = []
        if boxes is not None:
            for box, conf in zip(boxes, probs):
                if conf is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                detections.append(((x1, y1, x2 - x1, y2 - y1), float(conf)))

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
