from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

from app.events import broadcaster
from app.jobs import job_store
from app.storage import job_file

from src.name_tagging.final_name_assignment import assign_names
from src.pipeline.final_track_summary import build_track_summary

from .associate import associate_speakers
from .audio_extract import extract_audio
from .diarize import diarize_audio
from .emotion import infer_emotions
from .export_ui import export_ui
from .face_detect_track import detect_and_track
from .face_embed import embed_faces, embed_track_window
from .identity import enroll_identity, match_identity
from .transcribe import transcribe_and_attribute
from .utils import console, load_json, save_json


def _count_unique_speakers(segments: List[dict]) -> int:
    return len({str(s.get("speaker_id")) for s in segments if s.get("speaker_id")})


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "app").exists() or (parent / "pipeline").exists():
            return parent
    return here.parent


def _ensure_identity_store_files() -> None:
    """
    Always create identity_store/identities.json so it is never missing.
    """
    root = _project_root() / "identity_store"
    root.mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)

    identities_json = root / "identities.json"
    if not identities_json.exists():
        identities_json.write_text(
            json.dumps({"version": 1, "identities": []}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class _IdentityStoreAdapter:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def match(self, embedding_path: Path, exclude_person_ids: Optional[List[str]] = None) -> Dict:
        matched = match_identity(embedding_path, exclude_person_ids=exclude_person_ids or [])
        return {
            "status": matched.get("status", "unknown"),
            "score": float(matched.get("score", 0.0)),
            "margin": float(matched.get("margin", 0.0)),
            "identity_id": matched.get("person_id") or matched.get("candidate_person_id"),
            "person_id": matched.get("person_id"),
            "name": matched.get("name") if matched.get("status") == "matched" else matched.get("candidate_name"),
            "top_candidates": matched.get("top_candidates", []),
        }

    def enroll(self, name: str, embedding_path: Path, metadata: Dict) -> Dict:
        track_id = metadata.get("track_id") or Path(str(embedding_path)).stem
        association = {
            "job_id": metadata.get("job_id", self.job_id),
            "track_id": track_id,
            "speaker_id": metadata.get("speaker_id"),
            "name": name,
            "source": metadata.get("source", "self_intro"),
            "first_seen_ts": float(metadata.get("first_seen_ts", 0.0)),
        }
        return enroll_identity(
            self.job_id,
            track_id,
            name=name,
            association=association,
            merge_by_name=True,
            embedding_path=embedding_path,  # IMPORTANT
        )


def _ensure_identity_payload(track: Dict) -> Dict:
    identity = track.setdefault("identity", {})
    if not identity.get("name") or identity.get("name") == "Anonymous":
        identity["name"] = "Unknown"
    identity.setdefault("status", "unknown")
    identity.setdefault("person_id", None)
    identity.setdefault("score", 0.0)
    identity.setdefault("margin", 0.0)
    identity.setdefault("top_candidates", [])
    identity.setdefault("label_timeline", [])
    identity.setdefault("label_source", "none")
    return identity


def _append_identity_label(track: Dict, name: str, start_time: float, source: str, confidence: float = 1.0) -> None:
    identity = _ensure_identity_payload(track)
    timeline = identity.setdefault("label_timeline", [])
    safe_start = max(0.0, float(start_time))
    existing = next((entry for entry in timeline if float(entry.get("start", -1.0)) == safe_start), None)
    payload = {
        "start": safe_start,
        "name": name,
        "source": source,
        "confidence": max(0.0, min(1.0, float(confidence))),
    }
    if existing:
        existing.update(payload)
    else:
        timeline.append(payload)
        timeline.sort(key=lambda entry: float(entry.get("start", 0.0)))


def _publish(job_id: str, event: str, data: Dict) -> None:
    asyncio.run(broadcaster.publish(job_id, event, data))


def _tracks_overlap(track_a: Dict, track_b: Dict, slack: float = 0.5) -> bool:
    a0 = float(track_a.get("start", 0.0))
    a1 = float(track_a.get("end", a0))
    b0 = float(track_b.get("start", 0.0))
    b1 = float(track_b.get("end", b0))
    return max(a0, b0) <= min(a1, b1) + slack


def _track_avg_area(track: Dict) -> float:
    areas = []
    for bb in track.get("bboxes") or []:
        w = float(bb.get("w", 0.0))
        h = float(bb.get("h", 0.0))
        if w > 0 and h > 0:
            areas.append(w * h)
    if not areas:
        return 0.0
    return float(sum(areas) / len(areas))


def _track_reid_eligible(track: Dict) -> bool:
    bboxes = track.get("bboxes") or []
    if len(bboxes) < 2:
        return False
    if _track_avg_area(track) < 90 * 90:
        return False
    return True


def _enroll_from_named_segments(
    job_id: str,
    video_path: Path,
    tracks: List[Dict],
    assignment: Dict,
    identity_store: _IdentityStoreAdapter,
) -> None:
    """
    Only explicit self-intro segments can create memory.
    This is the part that fills identities.json and identity_store/embeddings/.
    """
    by_track = {str(t.get("track_id")): t for t in tracks}
    track_results = {str(t.get("track_id")): t for t in assignment.get("tracks", [])}
    tmp_dir = job_file(job_id, "embeddings") / "named_windows"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for seg in assignment.get("segment_assignments", []):
        self_names = seg.get("self_names") or []
        if not self_names:
            continue

        seg_name = str(self_names[0]).strip()
        if not seg_name or seg_name == "Unknown":
            continue

        track_id = str(seg.get("track_id") or "")
        if not track_id:
            continue

        track = by_track.get(track_id)
        result = track_results.get(track_id)
        if not track or not result:
            continue

        final_label = str(result.get("label") or "Unknown").strip()
        final_source = str(result.get("label_source") or "none").strip()

        # Only safe single-name self-intro tracks can be enrolled
        if final_label == "Unknown":
            continue
        if final_source != "self_intro":
            continue
        if final_label != seg_name:
            continue
        if not _track_reid_eligible(track):
            continue

        start_ts = float(seg.get("start_ts", 0.0))
        end_ts = float(seg.get("end_ts", start_ts))

        out_path = tmp_dir / f"{track_id}_{start_ts:.2f}_{end_ts:.2f}.npy"
        emb_path = embed_track_window(
            video_path=video_path,
            track=track,
            start_ts=start_ts,
            end_ts=end_ts,
            output_path=out_path,
            max_samples=6,
        )
        if not emb_path:
            continue

        try:
            identity_store.enroll(
                name=seg_name,
                embedding_path=emb_path,
                metadata={
                    "job_id": job_id,
                    "track_id": track_id,
                    "speaker_id": seg.get("speaker_id"),
                    "source": "self_intro_window",
                    "first_seen_ts": start_ts,
                },
            )
        except Exception as exc:
            console.log(f"[yellow]window-enroll failed for {track_id}/{seg_name}: {exc}[/yellow]")


def _apply_safe_reid(tracks: List[Dict], identity_store: _IdentityStoreAdapter) -> None:
    """
    Re-ID only after naming, and only for still-unknown tracks.
    Never overwrite transcript/self-intro names.
    """
    strong_named_tracks = []
    for tr in tracks:
        identity = tr.get("identity") or {}
        if identity.get("person_id") and str(identity.get("name")) not in {"", "Unknown"}:
            strong_named_tracks.append(tr)

    used_by_overlap: Dict[str, List[Dict]] = {}
    for tr in strong_named_tracks:
        pid = str((tr.get("identity") or {}).get("person_id"))
        if pid:
            used_by_overlap.setdefault(pid, []).append(tr)

    unknown_tracks = sorted(
        [tr for tr in tracks if str((tr.get("identity") or {}).get("name")) in {"", "Unknown"}],
        key=lambda t: (float(t.get("start", 0.0)), float(t.get("end", t.get("start", 0.0)))),
    )

    for tr in unknown_tracks:
        if not _track_reid_eligible(tr):
            continue

        emb_path = tr.get("embedding_path")
        if not emb_path:
            continue

        exclude_ids: List[str] = []
        for pid, pid_tracks in used_by_overlap.items():
            if any(_tracks_overlap(tr, named_tr, slack=0.5) for named_tr in pid_tracks):
                exclude_ids.append(pid)

        matched = identity_store.match(Path(str(emb_path)), exclude_person_ids=exclude_ids)

        tr.setdefault("identity", {})
        tr["identity"]["top_candidates"] = matched.get("top_candidates", [])
        tr["identity"]["margin"] = float(matched.get("margin", 0.0))

        if matched.get("status") != "matched":
            continue

        person_id = matched.get("person_id") or matched.get("identity_id")
        name = matched.get("name")
        if not person_id or not name:
            continue

        tr["identity"]["person_id"] = str(person_id)
        tr["identity"]["name"] = str(name)
        tr["identity"]["status"] = "matched"
        tr["identity"]["score"] = float(matched.get("score", 0.0))
        tr["identity"]["label_source"] = "identity_store"

        _append_identity_label(
            tr,
            str(name),
            float(tr.get("start", 0.0)),
            "identity_store",
            float(matched.get("score", 0.0)),
        )

        used_by_overlap.setdefault(str(person_id), []).append(tr)


def run_pipeline(job_id: str, video_path: Path) -> None:
    job_store.update_job(job_id, status="running", stage="extract_audio", progress=0)
    artifacts: Dict[str, str] = {}

    try:
        _ensure_identity_store_files()

        # 1) audio
        audio_path = job_file(job_id, "audio.wav")
        if not audio_path.exists():
            extract_audio(video_path, audio_path)
        artifacts["audio_wav"] = str(audio_path)
        job_store.update_job(job_id, stage="faces", progress=10, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "extract_audio", "progress": 10})

        # 2) face tracking
        faces_path = job_file(job_id, "faces_tracks.json")
        if not faces_path.exists():
            detect_and_track(video_path, faces_path)
        artifacts["faces_tracks"] = str(faces_path)
        tracks = load_json(faces_path).get("tracks", [])

        job_store.update_job(job_id, stage="identity", progress=30, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "faces", "progress": 30})

        # 3) embeddings
        embeddings_dir = job_file(job_id, "embeddings") / "face"
        embed_paths = embed_faces(video_path, tracks, embeddings_dir)

        identity_store = _IdentityStoreAdapter(job_id)

        # IMPORTANT: do not auto-label anything before name assignment
        for track in tracks:
            track_id = str(track.get("track_id"))
            emb_path = embed_paths.get(track_id)
            track["embedding_path"] = str(emb_path) if emb_path else None

            track["identity"] = {
                "status": "unknown",
                "person_id": None,
                "name": "Unknown",
                "score": 0.0,
                "margin": 0.0,
                "top_candidates": [],
                "label_timeline": [],
                "label_source": "none",
            }
            _ensure_identity_payload(track)

        # 4) emotion
        job_store.update_job(job_id, stage="emotion", progress=50, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "identity", "progress": 50})

        emotions_path = job_file(job_id, "emotions.json")
        emotions = load_json(emotions_path).get("tracks", {}) if emotions_path.exists() else infer_emotions(
            tracks, emotions_path, video_path
        )
        artifacts["emotions"] = str(emotions_path)

        for track in tracks:
            track["emotion"] = emotions.get(track["track_id"], {"dominant": "neutral", "timeline": []})

        # 5) diarization
        diar_path = job_file(job_id, "diarization.json")
        speakers: List[dict] = []
        if diar_path.exists():
            speakers = load_json(diar_path).get("segments", [])

        if not speakers or _count_unique_speakers(speakers) < 2:
            console.log("[yellow]Diarization missing or single-speaker; re-running diarization...[/yellow]")
            speakers = diarize_audio(audio_path, diar_path)
        artifacts["diarization"] = str(diar_path)

        # 6) transcript
        transcript_path = job_file(job_id, "transcript.json")
        bundle = transcribe_and_attribute(
            audio_path=audio_path,
            transcript_output_path=transcript_path,
            diarization_segments=speakers,
            diarization_output_path=diar_path,
            max_speakers=8,
        )
        speakers = bundle["diarization"]
        transcript_segments = bundle["transcript"]
        artifacts["transcript"] = str(transcript_path)

        # 7) associations
        job_store.update_job(job_id, stage="associate", progress=70, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "diarize", "progress": 70})

        assoc_path = job_file(job_id, "associations.json")
        associations = associate_speakers(tracks, transcript_segments, assoc_path)
        artifacts["associations"] = str(assoc_path)

        # 8) name tagging from real transcript segments
        assignment = assign_names(
            job_id=job_id,
            face_tracks=tracks,
            diarized_segments=[
                {
                    "speaker_id": seg.get("speaker_id"),
                    "start_ts": float(seg.get("start", 0.0)),
                    "end_ts": float(seg.get("end", seg.get("start", 0.0))),
                    "transcript_text": str(seg.get("text") or ""),
                }
                for seg in transcript_segments
            ],
            identity_store=identity_store,
            asd=None,
            config={
                "min_self_confidence": 0.40,
                "min_mention_confidence": 0.45,
                "auto_enroll_confidence": 0.60,
            },
        )

        # 9) explicit self-intro segments create memory
        _enroll_from_named_segments(
            job_id=job_id,
            video_path=video_path,
            tracks=tracks,
            assignment=assignment,
            identity_store=identity_store,
        )

        label_by_track = {item["track_id"]: item for item in assignment.get("tracks", [])}

        for track in tracks:
            resolved = label_by_track.get(track.get("track_id"), {})
            label = resolved.get("label", "Unknown")
            source = resolved.get("label_source", "none")
            confidence = float(resolved.get("confidence", 0.0))
            person_id = resolved.get("person_id")
            timeline = resolved.get("timeline", [])

            track.setdefault("identity", {})
            track["identity"]["label_timeline"] = timeline
            track["identity"]["label_source"] = source

            if source in {"timeline_ambiguous", "timeline_multi_name"}:
                track["identity"]["name"] = "Unknown"
                track["identity"]["status"] = "unknown"
                track["identity"]["score"] = 0.0
                track["identity"]["person_id"] = None
            elif label != "Unknown":
                track["identity"]["name"] = label
                track["identity"]["status"] = "matched"
                track["identity"]["score"] = confidence
                if person_id:
                    track["identity"]["person_id"] = person_id
                _append_identity_label(
                    track,
                    label,
                    float(resolved.get("first_seen_ts", 0.0)),
                    source,
                    confidence,
                )
            else:
                if not track["identity"].get("name"):
                    track["identity"]["name"] = "Unknown"
                if not track["identity"].get("status"):
                    track["identity"]["status"] = "unknown"

        # 10) only now try re-id on unknown tracks
        _apply_safe_reid(tracks, identity_store)

        save_json(
            assoc_path,
            {
                "associations": associations,
                "tracks": assignment.get("tracks", []),
                "segment_assignments": assignment.get("segment_assignments", []),
                "event_log": assignment.get("event_log", []),
            },
        )

        # 11) summary
        emotion_events = []
        for track_id, emotion_payload in emotions.items():
            for row in emotion_payload.get("timeline", []):
                emotion_events.append(
                    {
                        "track_id": track_id,
                        "start": float(row.get("start", 0.0)),
                        "ts": float(row.get("start", 0.0)),
                        "emotion": row.get("label", "neutral"),
                        "confidence": float(row.get("conf", 0.0)),
                    }
                )

        build_track_summary(
            job_id=job_id,
            face_tracks=tracks,
            emotion_events=emotion_events,
            diarized_segments=speakers,
            asr_segments=transcript_segments,
            identity_store=identity_store,
            config=None,
        )

        # 12) export
        job_store.update_job(job_id, stage="export", progress=90, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "associate", "progress": 90})

        ui_path = job_file(job_id, "ui.json")
        export_ui(ui_path, video_path, tracks, speakers, associations, artifacts)

        job_store.update_job(job_id, status="done", stage="export", progress=100, artifacts=artifacts)
        _publish(job_id, "job.done", {"ui_path": str(ui_path)})

    except Exception as exc:
        console.log(f"Pipeline failed: {exc}")
        job_store.update_job(
            job_id,
            status="error",
            stage="error",
            progress=100,
            error=str(exc),
            artifacts=artifacts,
        )
        _publish(job_id, "job.error", {"error": str(exc)})