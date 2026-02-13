from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.identity import match_identity
from pipeline.utils import save_json
from src.diarization.transcript_alignment import align_asr_to_diarization
from src.name_tagging.multilingual_name_extractor import extract_name_signals_from_segments


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _track_window(track: Dict[str, Any]) -> Tuple[float, float]:
    start = float(track.get("start", track.get("start_ts", 0.0)))
    end = float(track.get("end", track.get("end_ts", start)))
    return start, end


def _bbox_area_presence(track: Dict[str, Any], start: float, end: float) -> Tuple[float, float]:
    boxes = track.get("bboxes", [])
    relevant = [b for b in boxes if start <= float(b.get("t", -1.0)) <= end]
    if not relevant:
        return 0.0, 0.0
    avg_area = sum(float(b.get("w", 0.0)) * float(b.get("h", 0.0)) for b in relevant) / max(1, len(relevant))
    t0, t1 = _track_window(track)
    presence = _overlap(t0, t1, start, end) / max(0.001, end - start)
    return avg_area, presence


def _map_speaker_to_track(face_tracks: List[Dict[str, Any]], diarized_segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_speaker: Dict[str, List[Dict[str, Any]]] = {}
    for seg in diarized_segments:
        sid = str(seg.get("speaker_id") or "unknown")
        by_speaker.setdefault(sid, []).append(seg)

    mapping: Dict[str, Dict[str, Any]] = {}
    for speaker_id, segs in by_speaker.items():
        track_scores: Dict[str, float] = {}
        for seg in segs:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            for track in face_tracks:
                tid = track.get("track_id")
                if not tid:
                    continue
                area, presence = _bbox_area_presence(track, start, end)
                if presence <= 0:
                    continue
                track_scores[tid] = track_scores.get(tid, 0.0) + (0.65 * presence + 0.35 * (area / max(1.0, area)))
        if track_scores:
            best = max(track_scores.items(), key=lambda item: item[1])
            score_total = sum(track_scores.values())
            conf = best[1] / max(0.001, score_total)
            mapping[speaker_id] = {"track_id": best[0], "confidence": round(conf, 4)}
    return mapping


def _emotion_summary(track_id: str, emotion_events: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows = [r for r in emotion_events if r.get("track_id") == track_id]
    if not rows:
        return {"emotion": "neutral", "confidence": 0.0, "ts": None}, {"emotion": "neutral", "ratio": 0.0}
    current = max(rows, key=lambda r: float(r.get("ts", r.get("start", 0.0))))
    labels = [str(r.get("emotion", r.get("label", "neutral"))) for r in rows]
    counts = Counter(labels)
    dominant, count = max(counts.items(), key=lambda kv: kv[1])
    return (
        {
            "emotion": str(current.get("emotion", current.get("label", "neutral"))),
            "confidence": float(current.get("confidence", current.get("conf", 0.0))),
            "ts": float(current.get("ts", current.get("start", 0.0))),
        },
        {"emotion": dominant, "ratio": round(count / max(1, len(rows)), 4)},
    )


def _name_by_track(
    name_segments: List[Dict[str, Any]],
    speaker_track_map: Dict[str, Dict[str, Any]],
    face_tracks: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {str(t.get("track_id")): [] for t in face_tracks if t.get("track_id")}

    for seg in name_segments:
        sid = str(seg.get("speaker_id") or "unknown")
        speaker_tid = (speaker_track_map.get(sid) or {}).get("track_id")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        for signal in seg.get("signals", []):
            row = {
                "name": signal.get("name"),
                "type": signal.get("type"),
                "confidence": float(signal.get("confidence", 0.0)),
                "source_segment": [start, end],
            }
            if signal.get("type") == "self" and speaker_tid:
                out.setdefault(str(speaker_tid), []).append(row)
                continue
            if signal.get("type") == "mentioned":
                mid = (start + end) / 2.0
                best_tid = None
                best_overlap = 0.0
                for track in face_tracks:
                    tid = track.get("track_id")
                    if not tid or tid == speaker_tid:
                        continue
                    t0, t1 = _track_window(track)
                    ov = _overlap(t0, t1, mid - 2.0, mid + 2.0)
                    if ov > best_overlap:
                        best_overlap = ov
                        best_tid = tid
                if best_tid:
                    out.setdefault(str(best_tid), []).append(row)
    return out


def _identity_for_track(job_id: str, track_id: str, names: List[Dict[str, Any]], min_intro_conf: float, identity_store: Any = None) -> Dict[str, Any]:
    emb = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{track_id}.npy"
    if identity_store is not None and hasattr(identity_store, "match"):
        match = identity_store.match(emb)
    else:
        match = match_identity(emb)
    recognized = {
        "status": str(match.get("status", "unknown")),
        "identity_id": match.get("person_id") or match.get("candidate_person_id"),
        "name": match.get("name") if match.get("status") == "matched" else match.get("candidate_name"),
        "score": float(match.get("score", 0.0)),
    }
    best_self = max((n for n in names if n.get("type") == "self"), key=lambda n: n.get("confidence", 0.0), default=None)
    if best_self and float(best_self.get("confidence", 0.0)) >= min_intro_conf and recognized["status"] != "matched":
        recognized.update({"status": "unknown", "identity_id": None, "name": None})
    return recognized


def build_track_summary(
    job_id: str,
    face_tracks: List[Dict[str, Any]],
    emotion_events: List[Dict[str, Any]],
    diarized_segments: List[Dict[str, Any]],
    asr_segments: List[Dict[str, Any]],
    identity_store: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or {}
    min_name_conf = float(config.get("min_name_confidence", 0.55))
    min_intro_conf = float(config.get("min_intro_confidence", 0.8))

    aligned = align_asr_to_diarization(diarized_segments, asr_segments)
    name_segments = extract_name_signals_from_segments(aligned, {"min_name_confidence": min_name_conf})
    speaker_track_map = _map_speaker_to_track(face_tracks, diarized_segments)
    names_by_track = _name_by_track(name_segments, speaker_track_map, face_tracks)

    tracks: List[Dict[str, Any]] = []
    for track in face_tracks:
        track_id = str(track.get("track_id"))
        speaker_id = next((sid for sid, mapping in speaker_track_map.items() if mapping.get("track_id") == track_id), None)
        detected_names = names_by_track.get(track_id, [])
        language = next((seg.get("language") for seg in name_segments if str(seg.get("speaker_id")) == str(speaker_id)), "unknown")

        current_emotion, dominant_emotion = _emotion_summary(track_id, emotion_events)
        recognized_identity = _identity_for_track(job_id, track_id, detected_names, min_intro_conf, identity_store=identity_store)

        notes: List[str] = []
        if speaker_id is None:
            notes.append("speaker_unmapped")
        if not detected_names:
            notes.append("no_name_signals")

        tracks.append(
            {
                "track_id": track_id,
                "speaker_id": speaker_id,
                "language": language or "unknown",
                "detected_names": detected_names,
                "recognized_identity": recognized_identity,
                "current_emotion": current_emotion,
                "dominant_emotion": dominant_emotion,
                "notes": notes,
            }
        )

    summary = {"job_id": job_id, "tracks": tracks}
    out_path = Path("data") / "jobs" / job_id / "final_track_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, summary)
    return summary
