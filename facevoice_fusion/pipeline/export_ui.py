from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import save_json


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _emotion_at_time(track: Dict[str, Any], t: float) -> Dict[str, Any]:
    emotion = track.get("emotion") or {}
    timeline = emotion.get("timeline") or []

    best = None
    best_dt = 1e18
    for row in timeline:
        ts = _safe_float(row.get("start", row.get("ts", 0.0)))
        dt = abs(ts - t)
        if dt < best_dt:
            best_dt = dt
            best = row

    if best:
        return {
            "label": str(best.get("label", best.get("emotion", "neutral"))),
            "confidence": _safe_float(best.get("conf", best.get("confidence", 0.0))),
        }

    return {
        "label": str(emotion.get("dominant", "neutral")),
        "confidence": _safe_float(emotion.get("confidence", 0.0)),
    }


def _name_at_time(identity: Dict[str, Any], t: float) -> str:
    if not identity:
        return "Unknown"

    timeline = identity.get("label_timeline") or []
    if not timeline:
        nm = str(identity.get("name") or "Unknown").strip()
        return nm if nm else "Unknown"

    t = float(t)
    best = None
    best_conf = -1.0

    for entry in timeline:
        start = _safe_float(entry.get("start", -1e9))
        end = entry.get("end", None)
        end = _safe_float(end, 1e18) if end is not None else 1e18
        if start <= t <= end:
            conf = _safe_float(entry.get("confidence", 0.0))
            if conf > best_conf:
                best_conf = conf
                best = entry

    if best:
        nm = str(best.get("name") or "Unknown").strip()
        return nm if nm else "Unknown"

    prev = None
    prev_start = -1e18
    for entry in timeline:
        start = _safe_float(entry.get("start", -1e9))
        if start <= t and start > prev_start:
            prev = entry
            prev_start = start

    if prev:
        nm = str(prev.get("name") or "Unknown").strip()
        return nm if nm else "Unknown"

    nm = str(identity.get("name") or "Unknown").strip()
    return nm if nm else "Unknown"


def export_ui(
    ui_path: Path,
    video_path: Path,
    tracks: List[Dict[str, Any]],
    speakers: List[Dict[str, Any]],
    associations: List[Dict[str, Any]],
    artifacts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    artifacts = artifacts or {}

    ui_tracks: List[Dict[str, Any]] = []

    for tr in tracks:
        track_id = str(tr.get("track_id") or "")
        identity = tr.get("identity") or {}
        out_bboxes = []

        for bb in tr.get("bboxes") or []:
            t = _safe_float(bb.get("t", 0.0))
            emo = _emotion_at_time(tr, t)
            display_name = _name_at_time(identity, t)

            out_bboxes.append(
                {
                    "t": t,
                    "x": int(_safe_float(bb.get("x", 0))),
                    "y": int(_safe_float(bb.get("y", 0))),
                    "w": int(_safe_float(bb.get("w", 0))),
                    "h": int(_safe_float(bb.get("h", 0))),
                    "conf": _safe_float(bb.get("conf", 0.0)),
                    "display_name": display_name,
                    "display_emotion": emo["label"],
                    "display_emotion_confidence": emo["confidence"],
                }
            )

        ui_tracks.append(
            {
                "track_id": track_id,
                "start": _safe_float(tr.get("start", 0.0)),
                "end": _safe_float(tr.get("end", 0.0)),
                "identity": identity,
                "emotion": tr.get("emotion", {}),
                "bboxes": out_bboxes,
            }
        )

    payload = {
        "video_path": str(video_path),
        "artifacts": artifacts,
        "tracks": ui_tracks,
        "speakers": speakers,
        "associations": associations,
    }

    save_json(ui_path, payload)
    return payload