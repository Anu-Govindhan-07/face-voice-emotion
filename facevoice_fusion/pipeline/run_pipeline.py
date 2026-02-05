from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List

from app.events import broadcaster
from app.jobs import job_store
from app.storage import job_file
from .associate import associate_speakers
from .audio_extract import extract_audio
from .diarize import diarize_audio
from .emotion import infer_emotions
from .face_detect_track import detect_and_track
from .face_embed import embed_faces
from .identity import enroll_identity, match_identity, remember_identity_embedding
from .transcribe import infer_name_signals, transcribe_audio
from .utils import console, load_json, save_json


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
                remembered = remember_identity_embedding(
                    track_id=track["track_id"],
                    embedding_path=emb_path,
                    person_id=identity.get("person_id"),
                )
                if not identity.get("person_id") and remembered.get("person_id"):
                    identity["person_id"] = remembered["person_id"]
            else:
                identity = {"status": "unknown", "person_id": None, "name": "Unknown", "score": 0.0}
            track["identity"] = identity
            _publish(job_id, "track.identity", {"track_id": track["track_id"], "identity": identity})

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
        artifacts["transcript"] = str(transcript_path)

        job_store.update_job(job_id, stage="associate", progress=70, artifacts=artifacts)
        _publish(job_id, "job.progress", {"stage": "diarize", "progress": 70})

        assoc_path = job_file(job_id, "associations.json")
        name_signals = infer_name_signals(speakers, transcript_segments)
        speaker_names = name_signals.get("self", {})
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
            if inferred_name:
                track = next((item for item in tracks if item.get("track_id") == association.get("track_id")), None)
                if track:
                    if track.get("identity", {}).get("name") in {None, "", "Unknown"}:
                        track["identity"]["name"] = inferred_name
                    track["identity"]["status"] = "matched"
                    enrolled = enroll_identity(job_id, track["track_id"], inferred_name)
                    track["identity"]["person_id"] = enrolled.get("person_id")

            speaker_id = association.get("speaker_id")
            mentioned_name = mentioned_name_by_speaker.get(speaker_id)
            if mentioned_name:
                association["mentioned_name"] = mentioned_name
                target_track_id = association.get("track_id")
                speaker_track_id = speaker_to_track.get(speaker_id)
                if target_track_id and target_track_id == speaker_track_id:
                    target_track_id = next((
                        candidate_track_id
                        for candidate_speaker_id, candidate_track_id in speaker_to_track.items()
                        if candidate_speaker_id != speaker_id
                    ), target_track_id)
                target_track = next((item for item in tracks if item.get("track_id") == target_track_id), None)
                if target_track and target_track.get("identity", {}).get("name") in {None, "", "Unknown"}:
                    target_track["identity"]["name"] = mentioned_name
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
