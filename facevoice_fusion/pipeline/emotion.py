from __future__ import annotations

from typing import Dict, List

from .utils import console, save_json


EMOTION_LABEL = "neutral"


def infer_emotions(tracks: List[dict], output_path) -> Dict[str, dict]:
    console.log("Running emotion inference (stub)")
    emotions: Dict[str, dict] = {}
    for track in tracks:
        start = track.get("start", 0.0)
        end = track.get("end", 0.0)
        timeline = [{"start": start, "end": end, "label": EMOTION_LABEL, "conf": 0.3}]
        emotions[track["track_id"]] = {
            "dominant": EMOTION_LABEL,
            "timeline": timeline,
        }
    save_json(output_path, {"tracks": emotions})
    return emotions
