from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipeline.utils import save_json

_NAME_TOKEN = r"([A-Za-zÅÄÖåäö][A-Za-zÅÄÖåäö\-']{1,30})"
_SELF_PATTERNS = [
    re.compile(rf"\bi['’]?m\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bi am\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bmy name is\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bjag heter\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bmitt namn är\s+{_NAME_TOKEN}\b", re.IGNORECASE),
]
_MENTION_PATTERNS = [
    re.compile(rf"\b(?:this is|that is|det här är|detta är)\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\b(?:han heter|hon heter|he is|she is|called|named)\s+{_NAME_TOKEN}\b", re.IGNORECASE),
]
_STOPWORDS = {
    "jag", "du", "han", "hon", "det", "den", "mitt", "namn", "my", "name", "this", "that", "is", "är",
}


class _DefaultIdentityStore:
    def match(self, embedding_path: Path) -> Dict[str, Any]:
        return {"status": "unknown", "score": 0.0, "identity_id": None, "name": "Unknown"}

    def enroll(self, name: str, embedding_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "skipped", "name": name}


def _normalize_name(raw: str) -> Optional[str]:
    cleaned = re.sub(r"[^A-Za-zÅÄÖåäö\-']", "", (raw or "").strip())
    if len(cleaned) < 2:
        return None
    if cleaned.casefold() in _STOPWORDS:
        return None
    return cleaned[0].upper() + cleaned[1:].lower()


def _extract_name_signals(text: str) -> Dict[str, List[str]]:
    self_names: List[str] = []
    mentioned_names: List[str] = []
    for pattern in _SELF_PATTERNS:
        for match in pattern.finditer(text or ""):
            normalized = _normalize_name(match.group(1))
            if normalized and normalized not in self_names:
                self_names.append(normalized)
    for pattern in _MENTION_PATTERNS:
        for match in pattern.finditer(text or ""):
            normalized = _normalize_name(match.group(1))
            if normalized and normalized not in mentioned_names:
                mentioned_names.append(normalized)
    return {"self": self_names, "mentioned": mentioned_names}


def _track_start(track: Dict[str, Any]) -> float:
    return float(track.get("start_ts", track.get("start", 0.0)))


def _track_end(track: Dict[str, Any]) -> float:
    start = _track_start(track)
    return float(track.get("end_ts", track.get("end", start)))


def _window_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _boxes_for_track(track: Dict[str, Any]) -> List[Dict[str, Any]]:
    return track.get("bboxes") or track.get("bbox_timeline") or []


def _average_area(track: Dict[str, Any], start: float, end: float) -> float:
    boxes = [b for b in _boxes_for_track(track) if start <= float(b.get("t", -1.0)) <= end]
    if not boxes:
        return 0.0
    areas = [float(b.get("w", 0.0)) * float(b.get("h", 0.0)) for b in boxes]
    return sum(areas) / max(1, len(areas))


def _presence_ratio(track: Dict[str, Any], start: float, end: float) -> float:
    overlap = _window_overlap(_track_start(track), _track_end(track), start, end)
    return overlap / max(0.001, (end - start))


def _resolve_speaker_track(
    face_tracks: List[Dict[str, Any]],
    seg_start: float,
    seg_end: float,
    asd: Optional[Callable[..., Any] | Dict[float, Dict[str, float]]],
) -> Tuple[Optional[str], float, str]:
    if asd:
        scores: Dict[str, List[float]] = {}
        if callable(asd):
            maybe = asd((seg_start, seg_end))
            if isinstance(maybe, dict):
                for tid, score in maybe.items():
                    scores.setdefault(tid, []).append(float(score))
        elif isinstance(asd, dict):
            for ts, ts_scores in asd.items():
                if seg_start <= float(ts) <= seg_end:
                    for tid, score in ts_scores.items():
                        scores.setdefault(tid, []).append(float(score))
        if scores:
            averaged = {tid: (sum(vals) / len(vals)) for tid, vals in scores.items()}
            best = max(averaged.items(), key=lambda item: item[1])
            return best[0], max(0.0, min(1.0, best[1])), "asd"

    candidates = []
    for track in face_tracks:
        presence = _presence_ratio(track, seg_start, seg_end)
        if presence <= 0:
            continue
        area = _average_area(track, seg_start, seg_end)
        candidates.append((track.get("track_id"), presence, area))
    if not candidates:
        return None, 0.0, "none"

    max_area = max(item[2] for item in candidates) or 1.0
    ranked = []
    for track_id, presence, area in candidates:
        area_score = area / max_area
        confidence = (0.65 * presence) + (0.35 * area_score)
        ranked.append((track_id, confidence * 0.75))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[0][0], ranked[0][1], "fallback"


def _resolve_mention_track(
    face_tracks: List[Dict[str, Any]],
    mention_ts: float,
    speaker_track_id: Optional[str],
    temporal_window_seconds: float,
) -> Tuple[Optional[str], float, str]:
    candidates: List[Tuple[str, float]] = []
    for track in face_tracks:
        track_id = track.get("track_id")
        if not track_id or track_id == speaker_track_id:
            continue
        t_start, t_end = _track_start(track), _track_end(track)
        if t_end < mention_ts - temporal_window_seconds or t_start > mention_ts + temporal_window_seconds:
            continue
        proximity = max(0.0, 1.0 - abs(((t_start + t_end) / 2.0) - mention_ts) / max(0.1, temporal_window_seconds))
        recent_bonus = max(0.0, 1.0 - abs(t_start - mention_ts) / max(0.1, temporal_window_seconds))
        area = _average_area(track, mention_ts - 1.0, mention_ts + 1.0)
        candidates.append((track_id, 0.45 * proximity + 0.35 * recent_bonus + 0.20 * min(1.0, area / 25000.0)))

    if not candidates:
        return None, 0.0, "mentioned_out_of_scene"
    candidates.sort(key=lambda item: item[1], reverse=True)
    best_track, best_score = candidates[0]
    if len(candidates) > 1 and abs(candidates[0][1] - candidates[1][1]) < 0.08:
        return None, best_score, "ambiguous_mention"
    return best_track, min(1.0, best_score), "mention_ranked"


def assign_names(
    job_id: str,
    face_tracks: List[Dict[str, Any]],
    diarized_segments: List[Dict[str, Any]],
    identity_store: Any,
    asd: Optional[Callable[..., Any] | Dict[float, Dict[str, float]]] = None,
    config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    cfg = {
        "match_threshold": 0.75,
        "maybe_threshold": 0.60,
        "min_intro_confidence": 0.70,
        "temporal_window_seconds": 5.0,
        "auto_enroll_confidence": 0.90,
        "min_track_frames_for_enroll": 3,
    }
    if config:
        cfg.update(config)

    store = identity_store or _DefaultIdentityStore()
    assignments: Dict[str, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []

    for track in face_tracks:
        track_id = track.get("track_id")
        if not track_id:
            continue
        emb_path = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{track_id}.npy"
        label = "Unknown"
        source = "none"
        confidence = 0.0
        metadata: Dict[str, Any] = {}
        if emb_path.exists() and hasattr(store, "match"):
            match_result = store.match(emb_path)
            status = str(match_result.get("status", "unknown"))
            score = float(match_result.get("score", 0.0))
            if status == "matched" and score >= float(cfg["match_threshold"]):
                label = str(match_result.get("name") or "Unknown")
                source = "identity_store"
                confidence = score
                events.append({"ts": _track_start(track), "type": "match", "speaker_id": None, "name": label, "track_id": track_id, "confidence": confidence, "notes": "identity_store matched"})
            elif status == "maybe" and score >= float(cfg["maybe_threshold"]):
                metadata["candidate"] = {"name": match_result.get("name"), "score": score}
        assignments[track_id] = {
            "track_id": track_id,
            "label": label,
            "label_source": source,
            "confidence": confidence,
            "first_seen_ts": _track_start(track),
            "metadata": metadata,
        }

    segments = sorted(diarized_segments, key=lambda seg: float(seg.get("start_ts", seg.get("start", 0.0))))
    for segment in segments:
        text = str(segment.get("transcript_text", segment.get("text", "")))
        if not text.strip():
            continue
        speaker_id = segment.get("speaker_id")
        seg_start = float(segment.get("start_ts", segment.get("start", 0.0)))
        seg_end = float(segment.get("end_ts", segment.get("end", seg_start)))
        mention_ts = (seg_start + seg_end) / 2.0
        signals = _extract_name_signals(text)

        for name in signals["self"]:
            target_track_id, confidence, method = _resolve_speaker_track(face_tracks, seg_start, seg_end, asd)
            if not target_track_id or confidence < float(cfg["min_intro_confidence"]):
                events.append({"ts": mention_ts, "type": "self_intro", "speaker_id": speaker_id, "name": name, "track_id": None, "confidence": confidence, "notes": "unresolved_self_intro"})
                continue

            current = assignments[target_track_id]
            if current["label"] not in {"Unknown", name} and current["confidence"] >= confidence:
                events.append({"ts": mention_ts, "type": "self_intro", "speaker_id": speaker_id, "name": name, "track_id": target_track_id, "confidence": confidence, "notes": "conflict_kept_higher_confidence"})
                continue
            if current["label"] == name:
                current["confidence"] = min(1.0, max(current["confidence"], confidence) + 0.03)
            else:
                current.update({"label": name, "label_source": "self_intro", "confidence": confidence})
            events.append({"ts": mention_ts, "type": "self_intro", "speaker_id": speaker_id, "name": name, "track_id": target_track_id, "confidence": confidence, "notes": method})

            track_ref = next((t for t in face_tracks if t.get("track_id") == target_track_id), None)
            frame_count = len(_boxes_for_track(track_ref or {}))
            if confidence >= float(cfg["auto_enroll_confidence"]) and frame_count >= int(cfg["min_track_frames_for_enroll"]) and hasattr(store, "enroll"):
                emb_path = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{target_track_id}.npy"
                if emb_path.exists():
                    store.enroll(name, emb_path, {"job_id": job_id, "track_id": target_track_id, "speaker_id": speaker_id, "source": "self_intro"})

        for mentioned_name in signals["mentioned"]:
            speaker_track_id, _, _ = _resolve_speaker_track(face_tracks, seg_start, seg_end, asd)
            candidate_track_id, m_conf, reason = _resolve_mention_track(
                face_tracks,
                mention_ts,
                speaker_track_id=speaker_track_id,
                temporal_window_seconds=float(cfg["temporal_window_seconds"]),
            )
            if not candidate_track_id or m_conf < 0.62:
                events.append({"ts": mention_ts, "type": "mention", "speaker_id": speaker_id, "name": mentioned_name, "track_id": None, "confidence": m_conf, "notes": reason})
                continue

            current = assignments[candidate_track_id]
            if current["label"] == "Unknown":
                current.update({"label": mentioned_name, "label_source": "mention", "confidence": m_conf})
            elif current["label"] != mentioned_name:
                events.append({"ts": mention_ts, "type": "mention", "speaker_id": speaker_id, "name": mentioned_name, "track_id": candidate_track_id, "confidence": m_conf, "notes": "conflict_existing_label_kept"})
                continue
            events.append({"ts": mention_ts, "type": "mention", "speaker_id": speaker_id, "name": mentioned_name, "track_id": candidate_track_id, "confidence": m_conf, "notes": reason})

    result = {
        "tracks": list(assignments.values()),
        "event_log": events,
    }
    out_path = Path("data") / "jobs" / job_id / "associations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, result)
    return result
