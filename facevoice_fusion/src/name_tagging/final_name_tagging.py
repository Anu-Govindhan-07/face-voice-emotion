from __future__ import annotations

"""Final name detection + persistent re-identification stage.

Run with `run_final_name_tagging(...)` after diarization/ASR and face embeddings are ready.
This stage creates/updates:
- identity store: `<identity_store_dir>/identities.json` + embedding files
- per-job associations: `data/jobs/<job_id>/associations.json`
"""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    re.compile(rf"\b(?:han heter|hon heter)\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\b{_NAME_TOKEN}\s+(?:is|är)\b", re.IGNORECASE),
]
_STOPWORDS = {
    "jag",
    "du",
    "han",
    "hon",
    "det",
    "den",
    "mitt",
    "namn",
    "my",
    "name",
    "this",
    "that",
    "is",
    "är",
    "här",
    "detta",
}
_DEFAULT_CONFIG = {
    "MATCH_THRESHOLD": 0.45,
    "MAYBE_THRESHOLD": 0.35,
    "MIN_ASSIGN_CONFIDENCE": 0.70,
    "TEMPORAL_WINDOW_SECONDS": 5.0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_name(raw: str) -> Optional[str]:
    cleaned = re.sub(r"[^A-Za-zÅÄÖåäö\-']", "", (raw or "").strip())
    if len(cleaned) < 2:
        return None
    if cleaned.casefold() in _STOPWORDS:
        return None
    return cleaned[0].upper() + cleaned[1:].lower()


def parse_names_from_transcript(text: str) -> Dict[str, List[str]]:
    self_names: List[str] = []
    mentioned_names: List[str] = []
    content = text or ""

    for pattern in _SELF_PATTERNS:
        for match in pattern.finditer(content):
            name = _normalize_name(match.group(1))
            if name and name not in self_names:
                self_names.append(name)

    for pattern in _MENTION_PATTERNS:
        for match in pattern.finditer(content):
            name = _normalize_name(match.group(1))
            if name and name not in self_names and name not in mentioned_names:
                mentioned_names.append(name)

    return {"self_names": self_names, "mentioned_names": mentioned_names}


def _track_start(track: Dict[str, Any]) -> float:
    return float(track.get("start_ts", track.get("start", 0.0)))


def _track_end(track: Dict[str, Any]) -> float:
    start = _track_start(track)
    return float(track.get("end_ts", track.get("end", start)))


def _boxes_for_track(track: Dict[str, Any]) -> List[Dict[str, Any]]:
    return track.get("bboxes") or track.get("bbox_timeline") or []


def _window_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _presence_ratio(track: Dict[str, Any], start: float, end: float) -> float:
    overlap = _window_overlap(_track_start(track), _track_end(track), start, end)
    return overlap / max(0.001, (end - start))


def _average_area(track: Dict[str, Any], start: float, end: float) -> float:
    boxes = [b for b in _boxes_for_track(track) if start <= float(b.get("t", -1.0)) <= end]
    if not boxes:
        return 0.0
    areas = [float(b.get("w", 0.0)) * float(b.get("h", 0.0)) for b in boxes]
    return sum(areas) / len(areas)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_vec = np.asarray(a).astype(np.float32).reshape(-1)
    b_vec = np.asarray(b).astype(np.float32).reshape(-1)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def load_identity_store(identity_store_dir: str = "identity_store") -> Dict[str, Any]:
    store_dir = Path(identity_store_dir)
    identities_path = store_dir / "identities.json"
    embeddings_dir = store_dir / "embeddings"
    store_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    if identities_path.exists():
        data = json.loads(identities_path.read_text())
    else:
        data = {"identities": []}
    data.setdefault("identities", [])
    data["_paths"] = {
        "store_dir": str(store_dir),
        "identities_path": str(identities_path),
        "embeddings_dir": str(embeddings_dir),
    }
    return data


def save_identity_store(store: Dict[str, Any], identity_store_dir: str = "identity_store") -> None:
    store_dir = Path(identity_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    identities_path = store_dir / "identities.json"
    payload = {"identities": store.get("identities", [])}
    identities_path.write_text(json.dumps(payload, indent=2))


def _find_identity_by_name(store: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for identity in store.get("identities", []):
        if str(identity.get("name", "")).casefold() == name.casefold():
            return identity
    return None


def enroll_identity(
    name: str,
    embedding: np.ndarray,
    metadata: Dict[str, Any],
    identity_store_dir: str = "identity_store",
    store: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    loaded_store = store or load_identity_store(identity_store_dir)
    normalized = _normalize_name(name) or name

    identity = _find_identity_by_name(loaded_store, normalized)
    now = _utc_now()
    if identity is None:
        identity = {
            "identity_id": f"id_{uuid.uuid4().hex[:12]}",
            "name": normalized,
            "embedding_files": [],
            "created_at": now,
            "last_seen_at": now,
            "sources": [],
        }
        loaded_store["identities"].append(identity)

    emb_id = f"emb_{uuid.uuid4().hex[:12]}"
    emb_rel = Path("embeddings") / identity["identity_id"] / f"{emb_id}.npy"
    emb_abs = Path(identity_store_dir) / emb_rel
    emb_abs.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_abs, np.asarray(embedding, dtype=np.float32))

    identity["embedding_files"].append(str(emb_rel))
    identity["last_seen_at"] = now
    source_row = {
        "embedding_id": emb_id,
        "job_id": metadata.get("job_id"),
        "track_id": metadata.get("track_id"),
        "timestamp": metadata.get("timestamp"),
        "confidence": metadata.get("confidence"),
        "source": metadata.get("source", "self_intro"),
    }
    identity["sources"].append(source_row)

    save_identity_store(loaded_store, identity_store_dir=identity_store_dir)
    return {"identity_id": identity["identity_id"], "name": identity["name"], "embedding_file": str(emb_rel)}


def match_identity(
    embedding: np.ndarray,
    store: Dict[str, Any],
    identity_store_dir: str = "identity_store",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    best: Optional[Dict[str, Any]] = None
    for identity in store.get("identities", []):
        for rel in identity.get("embedding_files", []):
            emb_path = Path(identity_store_dir) / rel
            if not emb_path.exists():
                continue
            score = cosine_similarity(embedding, np.load(emb_path))
            if best is None or score > best["score"]:
                best = {
                    "identity_id": identity["identity_id"],
                    "name": identity["name"],
                    "score": score,
                    "embedding_file": rel,
                }

    if best is None:
        return {"status": "unknown", "score": 0.0, "identity_id": None, "name": "Unknown"}

    score = float(best["score"])
    if score >= float(cfg["MATCH_THRESHOLD"]):
        return {"status": "matched", **best}
    if score >= float(cfg["MAYBE_THRESHOLD"]):
        return {"status": "maybe", **best}
    return {"status": "unknown", "score": score, "identity_id": None, "name": "Unknown"}


def _resolve_speaker_track(
    face_tracks: List[Dict[str, Any]],
    seg_start: float,
    seg_end: float,
    asd: Optional[Any],
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
            averaged = {tid: sum(vals) / len(vals) for tid, vals in scores.items()}
            track_id, score = max(averaged.items(), key=lambda item: item[1])
            return track_id, min(1.0, max(0.0, float(score))), "asd"

    candidates: List[Tuple[str, float]] = []
    max_area = 1.0
    per_track = []
    for track in face_tracks:
        tid = track.get("track_id")
        if not tid:
            continue
        presence = _presence_ratio(track, seg_start, seg_end)
        if presence <= 0:
            continue
        area = _average_area(track, seg_start, seg_end)
        max_area = max(max_area, area)
        per_track.append((tid, presence, area))

    for tid, presence, area in per_track:
        area_score = area / max_area
        confidence = (0.7 * presence) + (0.3 * area_score)
        candidates.append((tid, 0.75 * confidence))

    if not candidates:
        return None, 0.0, "no_visible_speaker_candidate"

    candidates.sort(key=lambda item: item[1], reverse=True)
    top = candidates[0]
    if len(candidates) > 1 and abs(candidates[0][1] - candidates[1][1]) < 0.05:
        return None, top[1], "ambiguous_speaker_candidate"
    return top[0], top[1], "fallback_presence_area"


def _resolve_mention_track(
    face_tracks: List[Dict[str, Any]],
    mention_ts: float,
    speaker_track_id: Optional[str],
    temporal_window_seconds: float,
) -> Tuple[Optional[str], float, str]:
    candidates: List[Tuple[str, float]] = []
    for track in face_tracks:
        tid = track.get("track_id")
        if not tid or tid == speaker_track_id:
            continue
        t_start = _track_start(track)
        t_end = _track_end(track)
        if t_end < mention_ts - temporal_window_seconds or t_start > mention_ts + temporal_window_seconds:
            continue
        proximity = max(0.0, 1.0 - abs(((t_start + t_end) / 2.0) - mention_ts) / max(0.1, temporal_window_seconds))
        area = _average_area(track, mention_ts - 1.0, mention_ts + 1.0)
        score = (0.75 * proximity) + (0.25 * min(1.0, area / 20000.0))
        candidates.append((tid, score))

    if not candidates:
        return None, 0.0, "mentioned_out_of_scene"
    candidates.sort(key=lambda item: item[1], reverse=True)
    if len(candidates) > 1 and abs(candidates[0][1] - candidates[1][1]) < 0.08:
        return None, candidates[0][1], "ambiguous_mention"
    best = candidates[0]
    return best[0], min(1.0, best[1]), "mention_ranked"


def assign_name_to_track(
    assignments: Dict[str, Dict[str, Any]],
    track_id: str,
    name: str,
    label_source: str,
    confidence: float,
    notes: str,
    event_log: List[Dict[str, Any]],
    event_payload: Dict[str, Any],
) -> bool:
    if track_id not in assignments:
        return False
    current = assignments[track_id]
    if current["label"] not in {"Unknown", name} and current["confidence"] >= confidence:
        event_log.append({**event_payload, "track_id": track_id, "confidence": confidence, "notes": "conflict_kept_higher_confidence"})
        return False

    if current["label"] == name:
        current["confidence"] = max(current["confidence"], confidence)
    else:
        current.update(
            {
                "label": name,
                "label_source": label_source,
                "confidence": confidence,
                "notes": notes,
            }
        )
    event_log.append({**event_payload, "track_id": track_id, "confidence": confidence, "notes": notes})
    return True


def run_final_name_tagging(
    job_id: str,
    face_tracks: List[Dict[str, Any]],
    diarized_segments: List[Dict[str, Any]],
    identity_store_dir: str = "identity_store",
    config: dict | None = None,
    asd: Optional[Any] = None,
) -> dict:
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    store = load_identity_store(identity_store_dir)
    assignments: Dict[str, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []

    for track in face_tracks:
        track_id = track.get("track_id")
        if not track_id:
            continue
        emb_path = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{track_id}.npy"
        payload = {
            "track_id": track_id,
            "label": "Unknown",
            "label_source": "none",
            "confidence": 0.0,
            "identity_id": None,
            "notes": "no_match",
        }
        if emb_path.exists():
            match = match_identity(np.load(emb_path), store, identity_store_dir=identity_store_dir, config=cfg)
            status = match["status"]
            if status == "matched":
                payload.update(
                    {
                        "label": match["name"],
                        "label_source": "identity_store",
                        "confidence": match["score"],
                        "identity_id": match["identity_id"],
                        "notes": "matched",
                    }
                )
            elif status == "maybe":
                payload["notes"] = "maybe_match"
                payload["candidate_suggestions"] = [
                    {
                        "identity_id": match["identity_id"],
                        "name": match["name"],
                        "score": match["score"],
                    }
                ]
            events.append(
                {
                    "type": "embedding_match",
                    "track_id": track_id,
                    "status": status,
                    "score": match["score"],
                    "identity_id": match.get("identity_id"),
                    "name": match.get("name"),
                }
            )
        assignments[track_id] = payload

    segments = sorted(diarized_segments, key=lambda seg: float(seg.get("start_ts", seg.get("start", 0.0))))
    for segment in segments:
        text = str(segment.get("transcript_text", segment.get("text", "")))
        if not text.strip():
            continue
        seg_start = float(segment.get("start_ts", segment.get("start", 0.0)))
        seg_end = float(segment.get("end_ts", segment.get("end", seg_start)))
        seg_mid = (seg_start + seg_end) / 2.0
        speaker_id = segment.get("speaker_id")
        names = parse_names_from_transcript(text)

        for name in names["self_names"]:
            track_id, confidence, reason = _resolve_speaker_track(face_tracks, seg_start, seg_end, asd)
            event_base = {
                "type": "self_intro",
                "speaker_id": speaker_id,
                "name": name,
                "segment_start": seg_start,
                "segment_end": seg_end,
            }
            if not track_id or confidence < float(cfg["MIN_ASSIGN_CONFIDENCE"]):
                events.append({**event_base, "track_id": None, "confidence": confidence, "notes": "unresolved_self_intro"})
                continue

            if assign_name_to_track(
                assignments,
                track_id,
                name,
                "self_intro",
                confidence,
                reason,
                events,
                event_base,
            ):
                emb_path = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{track_id}.npy"
                if emb_path.exists():
                    enroll_result = enroll_identity(
                        name,
                        np.load(emb_path),
                        {
                            "job_id": job_id,
                            "track_id": track_id,
                            "timestamp": seg_mid,
                            "confidence": confidence,
                            "source": "self_intro",
                        },
                        identity_store_dir=identity_store_dir,
                        store=store,
                    )
                    assignments[track_id]["identity_id"] = enroll_result["identity_id"]

        for name in names["mentioned_names"]:
            speaker_track_id, _, _ = _resolve_speaker_track(face_tracks, seg_start, seg_end, asd)
            track_id, confidence, reason = _resolve_mention_track(
                face_tracks,
                mention_ts=seg_mid,
                speaker_track_id=speaker_track_id,
                temporal_window_seconds=float(cfg["TEMPORAL_WINDOW_SECONDS"]),
            )
            event_base = {
                "type": "mention",
                "speaker_id": speaker_id,
                "name": name,
                "segment_start": seg_start,
                "segment_end": seg_end,
            }
            if not track_id or confidence < float(cfg["MIN_ASSIGN_CONFIDENCE"]):
                events.append({**event_base, "track_id": None, "confidence": confidence, "notes": reason})
                continue
            assign_name_to_track(
                assignments,
                track_id,
                name,
                "mention",
                confidence,
                reason,
                events,
                event_base,
            )

    result = {
        "tracks": list(assignments.values()),
        "event_log": events,
    }
    out_path = Path("data") / "jobs" / job_id / "associations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result
