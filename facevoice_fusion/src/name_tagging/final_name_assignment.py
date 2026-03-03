from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


NAME_TOKEN = r"([A-Za-zÅÄÖåäö][A-Za-zÅÄÖåäö\-']{0,30}(?:\s+[A-Za-zÅÄÖåäö][A-Za-zÅÄÖåäö\-']{0,30})?)"

SELF_PATTERNS = [
    re.compile(rf"\bmy name is\s+{NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bi am\s+{NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bi['’]?m\s+{NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bjag heter\s+{NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bmitt namn är\s+{NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bhej[,\s]+jag heter\s+{NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\btjena[,\s]+jag heter\s+{NAME_TOKEN}\b", re.IGNORECASE),
]

MENTION_PATTERNS = [
    re.compile(rf"\b(?:this is|that is|it's|it is|meet)\s+{NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\b(?:det här är|detta är)\s+{NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\b(?:han heter|hon heter)\s+{NAME_TOKEN}\b", re.IGNORECASE),
]

SPLIT_MULTI = re.compile(r"\s+(?:and|och|&|und)\s+", re.IGNORECASE)

STOPWORDS = {
    "jag", "du", "han", "hon", "vi", "ni", "dom", "de", "det", "den", "här", "där",
    "hej", "tjena", "heter", "mitt", "namn", "är",
    "and", "or", "the", "a", "an", "this", "that", "is", "are", "name", "my", "im", "i'm", "am", "i",
}


def _normalize_name(candidate: str) -> Optional[str]:
    if not candidate:
        return None
    cleaned = candidate.strip(" .,!?:;\"'()[]{}")
    if not cleaned:
        return None
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return None

    out = []
    for t in tokens[:2]:
        if t.casefold() in STOPWORDS:
            return None
        out.append(t[:1].upper() + t[1:])
    return " ".join(out)


def _extract_self_names(text: str) -> List[str]:
    out: List[str] = []
    t = text or ""
    for pat in SELF_PATTERNS:
        for m in pat.finditer(t):
            raw = m.group(1)
            for piece in SPLIT_MULTI.split(raw):
                nm = _normalize_name(piece)
                if nm:
                    out.append(nm)
    # unique preserve order
    seen = set()
    final = []
    for n in out:
        if n not in seen:
            seen.add(n)
            final.append(n)
    return final


def _extract_mentioned_names(text: str) -> List[str]:
    out: List[str] = []
    t = text or ""
    for pat in MENTION_PATTERNS:
        for m in pat.finditer(t):
            raw = m.group(1)
            for piece in SPLIT_MULTI.split(raw):
                nm = _normalize_name(piece)
                if nm:
                    out.append(nm)
    seen = set()
    final = []
    for n in out:
        if n not in seen:
            seen.add(n)
            final.append(n)
    return final


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _track_duration(track: Dict[str, Any]) -> float:
    start = float(track.get("start", 0.0))
    end = float(track.get("end", start))
    return max(1e-6, end - start)


def _window_bboxes(track: Dict[str, Any], start_ts: float, end_ts: float) -> List[Dict[str, Any]]:
    boxes = []
    for bb in (track.get("bboxes") or []):
        t = bb.get("t")
        if t is None:
            continue
        t = float(t)
        if start_ts <= t <= end_ts:
            boxes.append(bb)
    boxes.sort(key=lambda b: float(b.get("t", 0.0)))
    return boxes


def _presence_seconds_from_bboxes(track: Dict[str, Any], start_ts: float, end_ts: float) -> float:
    boxes = _window_bboxes(track, start_ts, end_ts)
    if not boxes:
        return 0.0
    if len(boxes) == 1:
        return 0.20
    ts = [float(bb["t"]) for bb in boxes]
    presence = 0.0
    for i in range(1, len(ts)):
        presence += min(ts[i] - ts[i - 1], 0.40)
    return float(max(presence, 0.25))


def _bbox_area(bb: Dict[str, Any]) -> float:
    return float(bb.get("w", 0.0)) * float(bb.get("h", 0.0))


def _area_near_mid(track: Dict[str, Any], mid: float, tol: float = 0.75) -> float:
    best_area = 0.0
    best_dt = 1e18
    for bb in (track.get("bboxes") or []):
        t = bb.get("t")
        if t is None:
            continue
        t = float(t)
        dt = abs(t - mid)
        if dt <= tol and dt < best_dt:
            best_dt = dt
            best_area = _bbox_area(bb)
    return float(best_area)


def _best_track_for_segment(
    face_tracks: List[Dict[str, Any]],
    start_ts: float,
    end_ts: float,
    exclude_track_ids: Optional[set[str]] = None,
    window_pad: float = 0.80,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Choose best visible track for a segment:
    - expanded window to avoid missing faces at boundaries
    - prefer tracks that have a bbox near the segment mid
    - tie-break by mid-area, then presence
    """
    exclude_track_ids = exclude_track_ids or set()

    seg_start = max(0.0, float(start_ts) - window_pad)
    seg_end = float(end_ts) + window_pad
    mid = (float(start_ts) + float(end_ts)) / 2.0

    best = None
    best_key = (-1, -1.0, -1.0)  # (mid_present, mid_area, presence)
    best_presence = 0.0

    for tr in face_tracks:
        tid = str(tr.get("track_id") or "")
        if not tid or tid in exclude_track_ids:
            continue

        tr_start = float(tr.get("start", 0.0))
        tr_end = float(tr.get("end", tr_start))
        if _overlap(tr_start, tr_end, seg_start, seg_end) <= 0.0:
            continue

        presence = _presence_seconds_from_bboxes(tr, seg_start, seg_end)
        mid_area = _area_near_mid(tr, mid, tol=0.90)
        mid_present = 1 if mid_area > 0 else 0

        key = (mid_present, mid_area, presence)
        if key > best_key:
            best = tr
            best_key = key
            best_presence = presence

    return best, float(best_presence)


@dataclass
class AssignedLabel:
    label: str
    source: str
    confidence: float
    first_seen_ts: float
    person_id: Optional[str] = None


def _safe_enroll_track(
    job_id: str,
    track: Dict[str, Any],
    name: str,
    first_seen_ts: float,
    identity_store: Any,
    event_log: List[Dict[str, Any]],
) -> Optional[str]:
    embedding_path = track.get("embedding_path")
    if not embedding_path:
        event_log.append(
            {"type": "enroll_skipped_no_embedding", "track_id": track.get("track_id"), "name": name}
        )
        return None

    try:
        enrolled = identity_store.enroll(
            name=name,
            embedding_path=Path(str(embedding_path)),
            metadata={
                "job_id": job_id,
                "track_id": track.get("track_id"),
                "source": "self_intro",
                "speaker_id": None,
                "first_seen_ts": float(first_seen_ts),
            },
        )
        person_id = enrolled.get("person_id")
        event_log.append(
            {
                "type": "identity_enrolled",
                "track_id": track.get("track_id"),
                "name": name,
                "person_id": person_id,
                "status": enrolled.get("status"),
            }
        )
        return str(person_id) if person_id else None
    except Exception as exc:
        event_log.append(
            {"type": "identity_enroll_error", "track_id": track.get("track_id"), "name": name, "error": str(exc)}
        )
        return None


def assign_names(
    job_id: str,
    face_tracks: List[Dict[str, Any]],
    diarized_segments: List[Dict[str, Any]],
    identity_store: Any,
    asd: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    IMPORTANT behavior:
    - Always write per-segment label_timeline entries when a name is detected.
    - Even if a track has multiple names, keep label_timeline (UI can show correct name at time t).
    - Only enroll into identity store when a track is not multi-named.
    """
    config = config or {}
    min_self_conf = float(config.get("min_self_confidence", 0.40))

    label_timeline: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    self_votes: Dict[str, Counter] = defaultdict(Counter)
    first_seen: Dict[Tuple[str, str], float] = {}
    event_log: List[Dict[str, Any]] = []
    segment_assignments: List[Dict[str, Any]] = []

    # pass 1: assign self-intros to best visible track per segment
    for seg_idx, seg in enumerate(diarized_segments or []):
        start_ts = float(seg.get("start_ts", 0.0))
        end_ts = float(seg.get("end_ts", start_ts))
        if end_ts <= start_ts:
            continue

        text = str(seg.get("transcript_text") or "").strip()
        if not text:
            continue

        self_names = _extract_self_names(text)
        if not self_names:
            continue

        name = self_names[0]

        track, presence = _best_track_for_segment(face_tracks, start_ts, end_ts)
        if not track:
            event_log.append(
                {"type": "segment_no_face", "segment_index": seg_idx, "start_ts": start_ts, "end_ts": end_ts, "text": text[:140]}
            )
            continue

        track_id = str(track.get("track_id"))
        seg_dur = max(1e-6, end_ts - start_ts)
        coverage = min(1.0, presence / seg_dur)
        # strong signal because it is explicit self-intro
        conf = min(0.98, 0.70 + 0.25 * coverage)
        if conf < min_self_conf:
            continue

        self_votes[track_id][name] += 1
        first_seen.setdefault((track_id, name), start_ts)

        # CRITICAL: always keep timeline entry (even if later multi-name)
        label_timeline[track_id].append(
            {
                "start": start_ts,
                "end": end_ts,
                "name": name,
                "source": "self_intro",
                "confidence": conf,
                "segment_index": seg_idx,
                "text": text,
            }
        )

        segment_assignments.append(
            {
                "segment_index": seg_idx,
                "speaker_id": str(seg.get("speaker_id") or ""),
                "track_id": track_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "text": text,
                "self_names": self_names,
                "resolution": "self_intro_best_track",
            }
        )

    tracks_out: List[Dict[str, Any]] = []

    # pass 2: finalize track labels
    for tr in face_tracks:
        track_id = str(tr.get("track_id") or "")
        if not track_id:
            continue

        timeline = sorted(label_timeline.get(track_id, []), key=lambda x: float(x["start"]))
        unique_self_names = list(self_votes.get(track_id, {}).keys())

        # default final label
        final_name = "Unknown"
        final_source = "none"
        final_conf = 0.0
        final_ts = float(tr.get("start", 0.0))
        person_id: Optional[str] = None

        if len(unique_self_names) == 1:
            nm = unique_self_names[0]
            votes = int(self_votes[track_id][nm])
            final_name = nm
            final_source = "self_intro"
            final_conf = min(0.98, 0.82 + 0.05 * max(0, votes - 1))
            final_ts = float(first_seen.get((track_id, nm), final_ts))

            # safe enroll (only when not multi-named)
            person_id = _safe_enroll_track(job_id, tr, nm, final_ts, identity_store, event_log)

        elif len(unique_self_names) > 1:
            # Multi-name track: keep timeline, but do NOT enroll and keep final label Unknown
            event_log.append(
                {"type": "track_multiple_self_names", "track_id": track_id, "names": unique_self_names}
            )
            final_name = "Unknown"
            final_source = "timeline_multi_name"
            final_conf = 0.0

        tr.setdefault("identity", {})
        tr["identity"]["label_timeline"] = timeline
        tr["identity"]["label_source"] = final_source
        tr["identity"]["name"] = final_name
        tr["identity"]["status"] = "matched" if final_name != "Unknown" else "unknown"
        tr["identity"]["score"] = float(final_conf)
        if person_id:
            tr["identity"]["person_id"] = person_id

        tracks_out.append(
            {
                "track_id": track_id,
                "label": final_name,
                "label_source": final_source,
                "confidence": float(final_conf),
                "first_seen_ts": float(final_ts),
                "person_id": person_id,
                "timeline": timeline,
            }
        )

    return {
        "job_id": job_id,
        "tracks": tracks_out,
        "segment_assignments": segment_assignments,
        "event_log": event_log,
    }
