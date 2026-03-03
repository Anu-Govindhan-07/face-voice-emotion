from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .utils import save_json


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _segment_duration(seg_start: float, seg_end: float) -> float:
    return max(1e-6, float(seg_end) - float(seg_start))


def _track_duration(track: dict) -> float:
    start = float(track.get("start", 0.0))
    end = float(track.get("end", start))
    return max(1e-6, end - start)


def _window_bboxes(track: dict, seg_start: float, seg_end: float) -> List[dict]:
    out = []
    for bb in track.get("bboxes") or []:
        t = bb.get("t")
        if t is None:
            continue
        t = float(t)
        if seg_start <= t <= seg_end:
            out.append(bb)
    out.sort(key=lambda b: float(b.get("t", 0.0)))
    return out


def _track_presence_seconds(track: dict, seg_start: float, seg_end: float) -> float:
    boxes = _window_bboxes(track, seg_start, seg_end)
    if not boxes:
        return 0.0
    if len(boxes) == 1:
        return 0.20

    ts = [float(b["t"]) for b in boxes]
    presence = 0.0
    for i in range(1, len(ts)):
        presence += min(ts[i] - ts[i - 1], 0.40)
    return float(max(presence, 0.25))


def _mean_bbox_area(track: dict, seg_start: float, seg_end: float) -> float:
    boxes = _window_bboxes(track, seg_start, seg_end)
    if not boxes:
        return 0.0
    vals = [float(b.get("w", 0.0)) * float(b.get("h", 0.0)) for b in boxes]
    return float(sum(vals) / max(1, len(vals)))


def _has_bbox_near(track: dict, t: float, tol: float = 0.60) -> bool:
    lo, hi = t - tol, t + tol
    for bb in track.get("bboxes") or []:
        bt = bb.get("t")
        if bt is None:
            continue
        bt = float(bt)
        if lo <= bt <= hi:
            return True
    return False


def _score_track_for_segment(track: dict, seg_start: float, seg_end: float) -> Tuple[float, float]:
    if _overlap(float(track.get("start", 0.0)), float(track.get("end", 0.0)), seg_start, seg_end) <= 0.0:
        return -1.0, 0.0

    presence = _track_presence_seconds(track, seg_start, seg_end)
    if presence <= 0.0:
        return -1.0, 0.0

    seg_dur = _segment_duration(seg_start, seg_end)
    coverage = presence / seg_dur
    precision = presence / _track_duration(track)
    area = _mean_bbox_area(track, seg_start, seg_end)

    score = 0.70 * coverage + 0.30 * precision

    seg_mid = (seg_start + seg_end) / 2.0
    if _has_bbox_near(track, seg_mid, tol=0.75):
        score *= 1.25

    if area > 0:
        score *= 1.0 + min(0.20, area / 250000.0)

    return float(score), float(presence)


def associate_speakers(
    tracks: List[dict],
    speakers: List[dict],
    output_path,
    speaker_names: Dict[str, str] | None = None,
) -> List[dict]:
    associations: List[dict] = []
    speaker_names = speaker_names or {}

    ordered_segments = sorted(
        speakers or [],
        key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", s.get("start", 0.0)))),
    )

    for seg_idx, seg in enumerate(ordered_segments):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        speaker_id = str(seg.get("speaker_id") or "S1")
        text = str(seg.get("text") or "")

        best_track_id: Optional[str] = None
        best_score = -1.0
        best_presence = 0.0

        for tr in tracks:
            score, presence = _score_track_for_segment(tr, seg_start, seg_end)
            if score > best_score:
                best_score = score
                best_presence = presence
                best_track_id = tr.get("track_id")

        associations.append(
            {
                "segment_index": seg_idx,
                "speaker_id": speaker_id,
                "track_id": best_track_id if best_score > 0 else None,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "overlap_sec": float(best_presence if best_score > 0 else 0.0),
                "confidence": float(max(0.0, best_score) if best_score > 0 else 0.0),
                "inferred_name": speaker_names.get(speaker_id),
                "text": text,
            }
        )

    save_json(output_path, {"associations": associations})
    return associations