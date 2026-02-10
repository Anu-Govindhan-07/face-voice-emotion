from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from app.events import broadcaster
from app.jobs import job_store
from app.storage import job_file
from app.config import ALLOW_TRANSCRIPT_IDENTITY_ENROLL
from .associate import associate_speakers
from .audio_extract import extract_audio
from .diarize import diarize_audio
from .emotion import infer_emotions
from .face_detect_track import detect_and_track
from .face_embed import embed_faces
from .identity import enroll_identity, match_identity, remember_identity_embedding
from .transcribe import attribute_speakers_to_segments, infer_name_signals, transcribe_audio
from .utils import console, load_json, save_json


def _ensure_identity_payload(track: Dict) -> Dict:
    identity = track.setdefault("identity", {})
    if not identity.get("name") or identity.get("name") == "Unknown":
        identity["name"] = "Anonymous"
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


def _find_track(tracks: List[Dict], track_id: Optional[str]) -> Optional[Dict]:
    if not track_id:
        return None
    return next((track for track in tracks if track.get("track_id") == track_id), None)


def _track_visible_at(track: Dict, ts: float, tolerance: float = 0.5) -> bool:
    bboxes = track.get("bboxes") or []
    if not bboxes:
        return False
    return any(abs(float(box.get("t", -9999.0)) - ts) <= tolerance for box in bboxes)


def _resolve_mentioned_target_track(
    tracks: List[Dict],
    speaker_track_id: Optional[str],
    utterance_midpoint: float,
) -> Optional[Dict]:
    best_track = None
    best_distance = float("inf")
    for track in tracks:
        if track.get("track_id") == speaker_track_id:
            continue
        if not _track_visible_at(track, utterance_midpoint):
            continue
        distance = min(abs(float(box.get("t", utterance_midpoint)) - utterance_midpoint) for box in track.get("bboxes", []))
        if distance < best_distance:
            best_distance = distance
            best_track = track
    return best_track


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
        face_data = load_json(faces_path)
        tracks = face_data.get("tracks", [])
        for track in tracks:
            _publish(job_id, "track.created", {"track": track})

        job_store.update_job(job_id, stage="identity", progress=30, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "faces", "progress": 30})

        embeddings_dir = job_file(job_id, "embeddings") / "face"
        embed_paths = embed_faces(video_path, tracks, embeddings_dir)
        for track in tracks:
            emb_path = embed_paths.get(track["track_id"])
            if emb_path:
                identity = match_identity(emb_path)
                if identity.get("status") == "matched" and identity.get("person_id"):
                    remember_identity_embedding(
                        track_id=track["track_id"],
                        embedding_path=emb_path,
                        person_id=identity.get("person_id"),
                        allow_create=False,
                    )
                else:
                    identity["person_id"] = None
                    identity["name"] = "Anonymous"
            else:
                identity = {"status": "unknown", "person_id": None, "name": "Anonymous", "score": 0.0}
            track["identity"] = identity
            _ensure_identity_payload(track)
            if track["identity"].get("name") not in {None, "", "Unknown", "Anonymous"}:
                _append_identity_label(track, track["identity"]["name"], 0.0, "identity_store", confidence=track["identity"].get("score", 1.0))
            else:
                _append_identity_label(track, "Anonymous", 0.0, "default")
            _publish(job_id, "track.identity", {"track_id": track["track_id"], "identity": track["identity"]})

        job_store.update_job(job_id, stage="emotion", progress=50, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "identity", "progress": 50})

        emotions_path = job_file(job_id, "emotions.json")
        if emotions_path.exists():
            emotions = load_json(emotions_path).get("tracks", {})
        else:
            emotions = infer_emotions(tracks, emotions_path, video_path)
        artifacts["emotions"] = str(emotions_path)
        for track in tracks:
            emotion = emotions.get(track["track_id"], {"dominant": "neutral", "timeline": []})
            track["emotion"] = emotion
            _publish(job_id, "track.emotion", {"track_id": track["track_id"], "emotion_update": emotion})

        diar_path = job_file(job_id, "diarization.json")
        if diar_path.exists():
            speakers = load_json(diar_path).get("segments", [])
        else:
            speakers = diarize_audio(audio_path, diar_path)
        artifacts["diarization"] = str(diar_path)
        for segment in speakers:
            _publish(job_id, "speaker.segment", {"segment": segment})

        transcript_path = job_file(job_id, "transcript.json")
        if transcript_path.exists():
            transcript_segments = load_json(transcript_path).get("segments", [])
        else:
            transcript_segments = transcribe_audio(audio_path, transcript_path)
        transcript_segments = attribute_speakers_to_segments(speakers, transcript_segments)
        save_json(transcript_path, {"segments": transcript_segments})
        artifacts["transcript"] = str(transcript_path)

        job_store.update_job(job_id, stage="associate", progress=70, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "diarize", "progress": 70})

        assoc_path = job_file(job_id, "associations.json")
        name_signals = infer_name_signals(speakers, transcript_segments)
        speaker_names = name_signals.get("self", {})
        transcript_by_speaker = name_signals.get("speaker_segments", {})
        if assoc_path.exists():
            associations = load_json(assoc_path).get("associations", [])
            if speaker_names:
                for association in associations:
                    speaker_id = association.get("speaker_id")
                    if speaker_id and not association.get("inferred_name"):
                        association["inferred_name"] = speaker_names.get(speaker_id)
        else:
            associations = associate_speakers(tracks, speakers, assoc_path, speaker_names=speaker_names)
        artifacts["associations"] = str(assoc_path)
        mentioned_name_by_speaker = name_signals.get("mentioned", {})
        speaker_to_track = {assoc.get("speaker_id"): assoc.get("track_id") for assoc in associations if assoc.get("speaker_id")}

        for association in associations:
            inferred_name = association.get("inferred_name")
            if inferred_name and ALLOW_TRANSCRIPT_IDENTITY_ENROLL:
                track = _find_track(tracks, association.get("track_id"))
                if track and track.get("identity", {}).get("name") in {None, "", "Unknown", "Anonymous", inferred_name}:
                    _append_identity_label(track, inferred_name, max(0.0, float(track.get("start", 0.0))), "self_identification")
                    track["identity"]["status"] = "matched"
                    enrolled = enroll_identity(
                        job_id,
                        track["track_id"],
                        inferred_name,
                        association={
                            "job_id": job_id,
                            "track_id": track.get("track_id"),
                            "speaker_id": association.get("speaker_id"),
                            "name": inferred_name,
                            "source": "speaker_self_identification",
                        },
                    )
                    track["identity"]["person_id"] = enrolled.get("person_id")

            speaker_id = association.get("speaker_id")
            mentioned_name = mentioned_name_by_speaker.get(speaker_id)
            if mentioned_name:
                association["mentioned_name"] = mentioned_name
                if ALLOW_TRANSCRIPT_IDENTITY_ENROLL:
                    speaker_track_id = speaker_to_track.get(speaker_id)
                    speaker_utterance = transcript_by_speaker.get(speaker_id, {})
                    utterance_midpoint = float(speaker_utterance.get("mid", 0.0))
                    target_track = _resolve_mentioned_target_track(
                        tracks,
                        speaker_track_id=speaker_track_id,
                        utterance_midpoint=utterance_midpoint,
                    )
                    if target_track and target_track.get("identity", {}).get("name") in {None, "", "Unknown", "Anonymous"}:
                        _append_identity_label(target_track, mentioned_name, utterance_midpoint, "mentioned_by_speaker")
                        enrolled = enroll_identity(job_id, target_track["track_id"], mentioned_name)
                        target_track["identity"]["person_id"] = enrolled.get("person_id")
                        target_track["identity"]["status"] = "matched"

            _publish(job_id, "association.updated", {"association": association})

        save_json(assoc_path, {"associations": associations, "speaker_names": speaker_names})

        job_store.update_job(job_id, stage="export", progress=90, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "associate", "progress": 90})

        ui_path = job_file(job_id, "ui.json")
        export_payload = {
            "tracks": tracks,
            "speakers": speakers,
            "associations": associations,
        }
        save_json(job_file(job_id, "tracks_enriched.json"), export_payload)
        from .export_ui import export_ui

        if not ui_path.exists():
            export_ui(ui_path, video_path, tracks, speakers, associations, artifacts)
        job_store.update_job(job_id, status="done", stage="export", progress=100, artifacts=artifacts)
        _publish(job_id, "job.done", {"ui_path": str(ui_path)})
    except Exception as exc:
        console.log(f"Pipeline failed: {exc}")
        job_store.update_job(job_id, status="failed", error=str(exc))
        _publish(job_id, "job.failed", {"error": str(exc)})
