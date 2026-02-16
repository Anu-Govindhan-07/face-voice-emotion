from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List

from app.config import ALLOW_TRANSCRIPT_IDENTITY_ENROLL
from app.events import broadcaster
from app.jobs import job_store
from app.storage import job_file
from src.name_tagging.final_name_assignment import assign_names
from src.pipeline.final_track_summary import build_track_summary

from .associate import associate_speakers
from .audio_extract import extract_audio
from .diarize import diarize_audio
from .emotion import infer_emotions
from .face_detect_track import detect_and_track
from .face_embed import embed_faces
from .identity import enroll_identity, match_identity, remember_identity_embedding
from .transcribe import (
    attribute_speakers_to_segments,
    build_diarization_from_transcript_segments,
    infer_speakers_from_transcript_turns,
    transcribe_audio,
)
from .utils import console, load_json, save_json


class _IdentityStoreAdapter:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def match(self, embedding_path: Path) -> Dict:
        matched = match_identity(embedding_path)
        return {
            "status": matched.get("status", "unknown"),
            "score": float(matched.get("score", 0.0)),
            "identity_id": matched.get("person_id") or matched.get("candidate_person_id"),
            "name": matched.get("name") if matched.get("status") == "matched" else matched.get("candidate_name"),
        }

    def enroll(self, name: str, embedding_path: Path, metadata: Dict) -> Dict:
        track_id = metadata.get("track_id") or embedding_path.stem
        association = {
            "job_id": metadata.get("job_id", self.job_id),
            "track_id": track_id,
            "speaker_id": metadata.get("speaker_id"),
            "name": name,
            "source": metadata.get("source", "self_intro"),
        }
        return enroll_identity(self.job_id, track_id, name=name, association=association, merge_by_name=True)


def _ensure_identity_payload(track: Dict) -> Dict:
    identity = track.setdefault("identity", {})
    if not identity.get("name") or identity.get("name") == "Anonymous":
        identity["name"] = "Unknown"
    identity.setdefault("status", "unknown")
    identity.setdefault("person_id", None)
    identity.setdefault("score", 0.0)
    identity.setdefault("label_timeline", [])
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
    identity["name"] = name


def _publish(job_id: str, event: str, data: Dict) -> None:
    asyncio.run(broadcaster.publish(job_id, event, data))


def run_pipeline(job_id: str, video_path: Path) -> None:
    job_store.update_job(job_id, status="running", stage="extract_audio", progress=0)
    artifacts: Dict[str, str] = {}
    try:
        audio_path = job_file(job_id, "audio.wav")
        if not audio_path.exists():
            extract_audio(video_path, audio_path)
        artifacts["audio_wav"] = str(audio_path)
        job_store.update_job(job_id, stage="faces", progress=10, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "extract_audio", "progress": 10})

        faces_path = job_file(job_id, "faces_tracks.json")
        if not faces_path.exists():
            detect_and_track(video_path, faces_path)
        artifacts["faces_tracks"] = str(faces_path)
        tracks = load_json(faces_path).get("tracks", [])

        job_store.update_job(job_id, stage="identity", progress=30, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "faces", "progress": 30})

        embeddings_dir = job_file(job_id, "embeddings") / "face"
        embed_paths = embed_faces(video_path, tracks, embeddings_dir)
        for track in tracks:
            emb_path = embed_paths.get(track["track_id"])
            if emb_path:
                identity = match_identity(emb_path)
                if identity.get("status") == "matched" and identity.get("person_id"):
                    remember_identity_embedding(track_id=track["track_id"], embedding_path=emb_path, person_id=identity.get("person_id"), allow_create=False)
                else:
                    identity["person_id"] = None
                    identity["name"] = "Unknown"
            else:
                identity = {"status": "unknown", "person_id": None, "name": "Unknown", "score": 0.0}
            track["identity"] = identity
            _ensure_identity_payload(track)
            base_name = track["identity"].get("name") or "Unknown"
            _append_identity_label(track, base_name if base_name != "Anonymous" else "Unknown", 0.0, "identity_store", confidence=track["identity"].get("score", 0.0))
            _publish(job_id, "track.identity", {"track_id": track["track_id"], "identity": track["identity"]})

        job_store.update_job(job_id, stage="emotion", progress=50, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "identity", "progress": 50})

        emotions_path = job_file(job_id, "emotions.json")
        emotions = load_json(emotions_path).get("tracks", {}) if emotions_path.exists() else infer_emotions(tracks, emotions_path, video_path)
        artifacts["emotions"] = str(emotions_path)
        for track in tracks:
            track["emotion"] = emotions.get(track["track_id"], {"dominant": "neutral", "timeline": []})

        diar_path = job_file(job_id, "diarization.json")
        speakers = load_json(diar_path).get("segments", []) if diar_path.exists() else diarize_audio(audio_path, diar_path)
        artifacts["diarization"] = str(diar_path)

        transcript_path = job_file(job_id, "transcript.json")
        transcript_segments = load_json(transcript_path).get("segments", []) if transcript_path.exists() else transcribe_audio(audio_path, transcript_path)
        transcript_segments = attribute_speakers_to_segments(speakers, transcript_segments)

        speaker_ids = {str(seg.get("speaker_id") or "") for seg in speakers if seg.get("speaker_id")}
        if len(speaker_ids) <= 1 and len(transcript_segments) >= 2:
            inferred_segments = infer_speakers_from_transcript_turns(transcript_segments)
            inferred_speaker_ids = {str(seg.get("speaker_id") or "") for seg in inferred_segments if seg.get("speaker_id")}
            if len(inferred_speaker_ids) > 1:
                transcript_segments = inferred_segments
                speakers = build_diarization_from_transcript_segments(inferred_segments)
                save_json(diar_path, {"segments": speakers})

        save_json(transcript_path, {"segments": transcript_segments})
        artifacts["transcript"] = str(transcript_path)

        job_store.update_job(job_id, stage="associate", progress=70, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "diarize", "progress": 70})

        assoc_path = job_file(job_id, "associations.json")
        associations = associate_speakers(tracks, speakers, assoc_path)

        name_assignment_config = {"auto_enroll_confidence": 0.90}
        if not ALLOW_TRANSCRIPT_IDENTITY_ENROLL:
            name_assignment_config["auto_enroll_confidence"] = 1.01

        assignment = assign_names(
            job_id=job_id,
            face_tracks=tracks,
            diarized_segments=[
                {
                    "speaker_id": seg.get("speaker_id"),
                    "start_ts": float(seg.get("start", 0.0)),
                    "end_ts": float(seg.get("end", seg.get("start", 0.0))),
                    "transcript_text": seg.get("text", ""),
                }
                for seg in transcript_segments
            ],
            identity_store=_IdentityStoreAdapter(job_id),
            asd=None,
            config=name_assignment_config,
        )
        label_by_track = {item["track_id"]: item for item in assignment.get("tracks", [])}
        for track in tracks:
            resolved = label_by_track.get(track.get("track_id"), {})
            label = resolved.get("label", "Unknown")
            source = resolved.get("label_source", "none")
            confidence = float(resolved.get("confidence", 0.0))
            if label == "Unknown":
                track["identity"]["status"] = "unknown"
            else:
                track["identity"]["status"] = "matched" if source in {"identity_store", "self_intro"} else "maybe"
            track["identity"]["name"] = label
            _append_identity_label(track, label, float(resolved.get("first_seen_ts", 0.0)), source, confidence)

        save_json(
            assoc_path,
            {
                "associations": associations,
                "tracks": assignment.get("tracks", []),
                "event_log": assignment.get("event_log", []),
            },
        )
        artifacts["associations"] = str(assoc_path)

        emotion_events = []
        for track_id, emotion_payload in emotions.items():
            for row in emotion_payload.get("timeline", []):
                emotion_events.append({
                    "track_id": track_id,
                    "start": float(row.get("start", 0.0)),
                    "ts": float(row.get("start", 0.0)),
                    "emotion": row.get("label", "neutral"),
                    "confidence": float(row.get("conf", 0.0)),
                })

        build_track_summary(
            job_id=job_id,
            face_tracks=tracks,
            emotion_events=emotion_events,
            diarized_segments=speakers,
            asr_segments=transcript_segments,
            identity_store=_IdentityStoreAdapter(job_id),
            config=None,
        )
        final_summary_path = job_file(job_id, "final_track_summary.json")
        artifacts["final_track_summary"] = str(final_summary_path)

        job_store.update_job(job_id, stage="export", progress=90, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "associate", "progress": 90})

        ui_path = job_file(job_id, "ui.json")
        save_json(job_file(job_id, "tracks_enriched.json"), {"tracks": tracks, "speakers": speakers, "associations": associations})
        from .export_ui import export_ui

        if not ui_path.exists():
            export_ui(ui_path, video_path, tracks, speakers, associations, artifacts)
        job_store.update_job(job_id, status="done", stage="export", progress=100, artifacts=artifacts)
        _publish(job_id, "job.done", {"ui_path": str(ui_path)})
    except Exception as exc:
        console.log(f"Pipeline failed: {exc}")
        job_store.update_job(job_id, status="failed", error=str(exc))
        _publish(job_id, "job.failed", {"error": str(exc)})
