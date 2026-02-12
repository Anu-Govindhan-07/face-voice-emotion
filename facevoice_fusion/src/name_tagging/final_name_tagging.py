from __future__ import annotations

"""Final name detection + persistent re-identification stage.

Run with `run_final_name_tagging(...)` after diarization/ASR and face embeddings are ready.
This stage creates/updates:
- identity store: `<identity_store_dir>/identities.json` + embedding files
- per-job associations: `data/jobs/<job_id>/associations.json`
"""

import json
import logging
import re
import uuid
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

_DEFAULT_NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
_MIN_SEGMENT_CHARS = 3
_SELF_CONTEXT_WINDOW = 28
_SELF_KEYWORDS = {
    "i am",
    "i'm",
    "my name",
    "name is",
    "jag heter",
    "mitt namn",
    "je m'appelle",
    "me llamo",
    "mi nombre",
    "ich bin",
    "mein name",
    "mi chiamo",
    "sou eu",
    "eu sou",
    "меня зовут",
    "我叫",
    "私の名前",
    "내 이름",
    "benim adım",
}
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
    "hello",
    "hej",
    "thanks",
    "tack",
}
_COMMON_FALSE_POSITIVES = {
    "unknown",
    "speaker",
    "audio",
    "video",
    "meeting",
    "interview",
    "person",
}
_DEFAULT_CONFIG = {
    "MATCH_THRESHOLD": 0.45,
    "MAYBE_THRESHOLD": 0.35,
    "MIN_ASSIGN_CONFIDENCE": 0.70,
    "TEMPORAL_WINDOW_SECONDS": 5.0,
    "SPEAKER_HINT_CONFIDENCE": 0.82,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_name(raw: str) -> Optional[str]:
    value = (raw or "").strip().strip(".,!?;:\"'`()[]{}")
    cleaned = "".join(ch for ch in value if ch.isalpha() or ch in "-'")
    if len(cleaned) < 2:
        return None
    normalized = unicodedata.normalize("NFKC", cleaned)
    if normalized.casefold() in _STOPWORDS or normalized.casefold() in _COMMON_FALSE_POSITIVES:
        return None
    return "-".join(part[:1].upper() + part[1:].lower() for part in normalized.split("-"))


class MultilingualNameDetector:
    def __init__(self, model_name: str = _DEFAULT_NER_MODEL) -> None:
        self.model_name = model_name
        self._ner = None
        self._ner_error: Optional[str] = None

    def _get_ner(self):
        if self._ner is not None:
            return self._ner
        if self._ner_error:
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self._ner = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
            logger.info("Loaded multilingual NER model: %s", self.model_name)
        except Exception as exc:
            self._ner_error = str(exc)
            logger.warning("Could not load multilingual NER model %s: %s", self.model_name, exc)
            self._ner = None
        return self._ner

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        ner = self._get_ner()
        if not text or len(text.strip()) < _MIN_SEGMENT_CHARS or ner is None:
            return []
        try:
            entities = ner(text)
        except Exception as exc:
            logger.warning("NER inference failed: %s", exc)
            return []

        out: List[Dict[str, Any]] = []
        for entity in entities:
            label = str(entity.get("entity_group") or entity.get("entity") or "")
            if "PER" not in label.upper() and "PERSON" not in label.upper():
                continue
            start = int(entity.get("start", 0))
            end = int(entity.get("end", start))
            raw_name = str(entity.get("word", ""))
            normalized_name = _normalize_name(raw_name)
            if not normalized_name:
                continue
            out.append({"name": normalized_name, "start": start, "end": end, "confidence": float(entity.get("score", 0.0))})
        return out


_NAME_DETECTOR = MultilingualNameDetector()


def _is_self_identification(text: str, start: int) -> bool:
    lower = text.casefold()
    context = lower[max(0, start - _SELF_CONTEXT_WINDOW) : start]
    return any(keyword in context for keyword in _SELF_KEYWORDS)


def _fallback_regex_entities(text: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    context_patterns = [
        re.compile(r"\b(?:i am|i['’]m|my name is|jag heter|mitt namn är|je m'appelle|me llamo|mi nombre es|ich bin|mein name ist)\s+([^\W\d_][^\s,.;:!?]{1,30})", re.IGNORECASE),
        re.compile(r"\b(?:this is|that is|det här är|detta är|han heter|hon heter)\s+([^\W\d_][^\s,.;:!?]{1,30})", re.IGNORECASE),
    ]
    for pattern in context_patterns:
        for match in pattern.finditer(text):
            name = _normalize_name(match.group(1))
            if name:
                candidates.append({"name": name, "start": match.start(1), "end": match.end(1), "confidence": 0.42})

    for match in re.finditer(r"\b([^\W\d_][^\W\d_\-']{1,30}(?:[-'][^\W\d_]{1,30})?)\b", text, flags=re.UNICODE):
        token = match.group(1)
        if not token or not token[0].isupper():
            continue
        name = _normalize_name(token)
        if name:
            candidates.append({"name": name, "start": match.start(1), "end": match.end(1), "confidence": 0.35})
    return candidates


def detect_names_from_segments(diarized_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    for segment in diarized_segments:
        text = str(segment.get("transcript_text", segment.get("text", "")))
        speaker_id = str(segment.get("speaker_id") or "")
        start_time = float(segment.get("start_ts", segment.get("start", 0.0)))
        end_time = float(segment.get("end_ts", segment.get("end", start_time)))

        detections = _NAME_DETECTOR.extract_entities(text)
        if not detections:
            detections = _fallback_regex_entities(text)

        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in detections:
            name = item["name"]
            entity_type = "self" if _is_self_identification(text, int(item.get("start", 0))) else "mentioned"
            key = (name.casefold(), entity_type)
            curr = merged.get(key)
            if curr is None or float(item.get("confidence", 0.0)) > curr["confidence"]:
                merged[key] = {"name": name, "type": entity_type, "confidence": round(float(item.get("confidence", 0.0)), 4)}

        detected_names = sorted(merged.values(), key=lambda row: row["confidence"], reverse=True)
        logger.info(
            "Name detection segment speaker=%s [%.2f-%.2f] text=%r names=%s",
            speaker_id or "unknown",
            start_time,
            end_time,
            text,
            detected_names,
        )
        signals.append(
            {
                "speaker_id": speaker_id,
                "start_time": start_time,
                "end_time": end_time,
                "detected_names": detected_names,
                "transcript_text": text,
            }
        )
    return signals


def parse_names_from_transcript(text: str) -> Dict[str, List[str]]:
    parsed = detect_names_from_segments([{"speaker_id": "", "start": 0.0, "end": 0.0, "text": text or ""}])[0]["detected_names"]
    self_names = [row["name"] for row in parsed if row["type"] == "self"]
    mentioned_names = [row["name"] for row in parsed if row["type"] == "mentioned"]
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
    identity["sources"].append(
        {
            "embedding_id": emb_id,
            "job_id": metadata.get("job_id"),
            "track_id": metadata.get("track_id"),
            "timestamp": metadata.get("timestamp"),
            "confidence": metadata.get("confidence"),
            "source": metadata.get("source", "self_intro"),
        }
    )

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

    if best["score"] >= float(cfg["MATCH_THRESHOLD"]):
        return {"status": "matched", **best}
    if best["score"] >= float(cfg["MAYBE_THRESHOLD"]):
        return {"status": "maybe", **best}
    return {"status": "unknown", "score": float(best["score"]), "identity_id": None, "name": "Unknown"}


def _build_speaker_track_hints(
    face_tracks: List[Dict[str, Any]], diarized_segments: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Infer a stable speaker->track map from full timeline overlap when ASD is unavailable."""
    by_speaker: Dict[str, Dict[str, float]] = {}
    for segment in diarized_segments:
        speaker_id = str(segment.get("speaker_id") or "")
        if not speaker_id:
            continue
        seg_start = float(segment.get("start_ts", segment.get("start", 0.0)))
        seg_end = float(segment.get("end_ts", segment.get("end", seg_start)))
        if seg_end <= seg_start:
            continue

        max_area = max((_average_area(track, seg_start, seg_end) for track in face_tracks), default=1.0) or 1.0
        scores = by_speaker.setdefault(speaker_id, {})
        for track in face_tracks:
            track_id = track.get("track_id")
            if not track_id:
                continue
            overlap = _window_overlap(_track_start(track), _track_end(track), seg_start, seg_end)
            if overlap <= 0:
                continue
            area_score = _average_area(track, seg_start, seg_end) / max_area
            scores[track_id] = scores.get(track_id, 0.0) + (overlap * (1.0 + 0.3 * area_score))

    hints: Dict[str, Dict[str, Any]] = {}
    for speaker_id, scores in by_speaker.items():
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_track, top_score = ranked[0]
        total = sum(scores.values()) or 1.0
        confidence = top_score / total
        hints[speaker_id] = {"track_id": top_track, "confidence": confidence}
    return hints


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
        current.update({"label": name, "label_source": label_source, "confidence": confidence, "notes": notes})
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
    speaker_hints = _build_speaker_track_hints(face_tracks, diarized_segments) if not asd else {}

    for track in face_tracks:
        track_id = track.get("track_id")
        if not track_id:
            continue
        emb_path = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{track_id}.npy"
        payload = {"track_id": track_id, "label": "Unknown", "label_source": "none", "confidence": 0.0, "identity_id": None, "notes": "no_match"}
        if emb_path.exists():
            match = match_identity(np.load(emb_path), store, identity_store_dir=identity_store_dir, config=cfg)
            status = match["status"]
            if status == "matched":
                payload.update({"label": match["name"], "label_source": "identity_store", "confidence": match["score"], "identity_id": match["identity_id"], "notes": "matched"})
            elif status == "maybe":
                payload["notes"] = "maybe_match"
                payload["candidate_suggestions"] = [{"identity_id": match["identity_id"], "name": match["name"], "score": match["score"]}]
            events.append({"type": "embedding_match", "track_id": track_id, "status": status, "score": match["score"], "identity_id": match.get("identity_id"), "name": match.get("name")})
        assignments[track_id] = payload

    name_signals = detect_names_from_segments(diarized_segments)
    segments = sorted(name_signals, key=lambda seg: float(seg.get("start_time", 0.0)))
    for segment in segments:
        text = str(segment.get("transcript_text", ""))
        if not text.strip():
            continue
        seg_start = float(segment.get("start_time", 0.0))
        seg_end = float(segment.get("end_time", seg_start))
        seg_mid = (seg_start + seg_end) / 2.0
        speaker_id = str(segment.get("speaker_id") or "")
        names = segment.get("detected_names", [])

        for detection in names:
            if detection.get("type") != "self":
                continue
            name = str(detection.get("name") or "")
            track_id, confidence, reason = _resolve_speaker_track(face_tracks, seg_start, seg_end, asd)
            confidence = max(confidence, float(detection.get("confidence", 0.0)))
            hint = speaker_hints.get(speaker_id)
            if (not track_id or confidence < float(cfg["MIN_ASSIGN_CONFIDENCE"])) and hint and hint["confidence"] >= float(cfg["SPEAKER_HINT_CONFIDENCE"]):
                track_id = hint["track_id"]
                confidence = max(confidence, min(0.95, hint["confidence"]))
                reason = "speaker_timeline_hint"

            event_base = {"type": "self_intro", "speaker_id": speaker_id or None, "name": name, "segment_start": seg_start, "segment_end": seg_end}
            if not track_id or confidence < float(cfg["MIN_ASSIGN_CONFIDENCE"]):
                events.append({**event_base, "track_id": None, "confidence": confidence, "notes": "unresolved_self_intro"})
                continue

            if assign_name_to_track(assignments, track_id, name, "self_intro", confidence, reason, events, event_base):
                emb_path = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{track_id}.npy"
                if emb_path.exists():
                    enrolled = enroll_identity(
                        name,
                        np.load(emb_path),
                        {"job_id": job_id, "track_id": track_id, "timestamp": seg_mid, "confidence": confidence, "source": "self_intro"},
                        identity_store_dir=identity_store_dir,
                        store=store,
                    )
                    assignments[track_id]["identity_id"] = enrolled["identity_id"]

        for detection in names:
            if detection.get("type") != "mentioned":
                continue
            name = str(detection.get("name") or "")
            speaker_track_id, _, _ = _resolve_speaker_track(face_tracks, seg_start, seg_end, asd)
            hint = speaker_hints.get(speaker_id)
            if hint and hint["confidence"] >= float(cfg["SPEAKER_HINT_CONFIDENCE"]):
                speaker_track_id = hint["track_id"]
            track_id, confidence, reason = _resolve_mention_track(
                face_tracks,
                mention_ts=seg_mid,
                speaker_track_id=speaker_track_id,
                temporal_window_seconds=float(cfg["TEMPORAL_WINDOW_SECONDS"]),
            )
            confidence = max(confidence, float(detection.get("confidence", 0.0)))
            event_base = {"type": "mention", "speaker_id": speaker_id or None, "name": name, "segment_start": seg_start, "segment_end": seg_end}
            if not track_id or confidence < float(cfg["MIN_ASSIGN_CONFIDENCE"]):
                events.append({**event_base, "track_id": None, "confidence": confidence, "notes": reason})
                continue
            assign_name_to_track(assignments, track_id, name, "mention", confidence, reason, events, event_base)

    result = {"tracks": list(assignments.values()), "event_log": events}
    out_path = Path("data") / "jobs" / job_id / "associations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result
