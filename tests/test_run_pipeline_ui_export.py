from __future__ import annotations

import json
import sys
import types
import importlib.machinery
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

# run_pipeline imports emotion.py which imports cv2; stub it for test environments without libGL.
if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    sys.modules["cv2"] = cv2_stub

from app.jobs import job_store
from pipeline import run_pipeline as run_pipeline_module


def test_run_pipeline_rebuilds_diarization_from_transcript_and_overwrites_ui(tmp_path, monkeypatch):
    job_id = "job_test"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Seed artifacts so pipeline reuses them instead of invoking heavy models.
    (job_dir / "audio.wav").write_bytes(b"RIFF")
    (job_dir / "faces_tracks.json").write_text(json.dumps({"tracks": [{"track_id": "T1"}]}))
    (job_dir / "emotions.json").write_text(json.dumps({"tracks": {"T1": {"dominant": "neutral", "timeline": []}}}))
    (job_dir / "diarization.json").write_text(
        json.dumps({"segments": [{"speaker_id": "S1", "start": 0.0, "end": 6.0}]})
    )
    (job_dir / "transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 3.0, "text": "hello", "speaker_id": "S1"},
                    {"start": 3.0, "end": 6.0, "text": "hi", "speaker_id": "S2"},
                ]
            }
        )
    )
    # Existing stale UI should be replaced.
    (job_dir / "ui.json").write_text(json.dumps({"speakers": [{"speaker_id": "S1"}], "stale": True}))

    monkeypatch.setattr(run_pipeline_module, "job_file", lambda _job_id, name: job_dir / name)
    monkeypatch.setattr(run_pipeline_module, "_publish", lambda *_args, **_kwargs: None)

    def _update_job(job_id: str, **_kwargs):
        return None

    monkeypatch.setattr(job_store, "update_job", _update_job)

    monkeypatch.setattr(run_pipeline_module, "extract_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_pipeline_module, "detect_and_track", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_pipeline_module, "embed_faces", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_pipeline_module, "match_identity", lambda *_args, **_kwargs: {"status": "unknown", "name": "Unknown", "score": 0.0})
    monkeypatch.setattr(run_pipeline_module, "remember_identity_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_pipeline_module, "infer_emotions", lambda *_args, **_kwargs: {"T1": {"dominant": "neutral", "timeline": []}})
    monkeypatch.setattr(run_pipeline_module, "diarize_audio", lambda *_args, **_kwargs: [{"speaker_id": "S1", "start": 0.0, "end": 6.0}])
    def _transcribe_and_attribute(**_kwargs):
        diarization = [
            {"speaker_id": "S1", "start": 0.0, "end": 3.0},
            {"speaker_id": "S2", "start": 3.0, "end": 6.0},
        ]
        transcript = [
            {"start": 0.0, "end": 3.0, "text": "hello", "speaker_id": "S1"},
            {"start": 3.0, "end": 6.0, "text": "hi", "speaker_id": "S2"},
        ]
        (job_dir / "diarization.json").write_text(json.dumps({"segments": diarization}))
        (job_dir / "transcript.json").write_text(json.dumps({"segments": transcript}))
        return {"diarization": diarization, "transcript": transcript}

    monkeypatch.setattr(run_pipeline_module, "transcribe_and_attribute", _transcribe_and_attribute)
    monkeypatch.setattr(run_pipeline_module, "associate_speakers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(run_pipeline_module, "assign_names", lambda **_kwargs: {"tracks": [], "event_log": []})
    monkeypatch.setattr(run_pipeline_module, "build_track_summary", lambda **_kwargs: None)

    captured = {}

    def _export_ui(output_path, _video_path, _tracks, speakers, _associations, _artifacts):
        captured["speakers"] = speakers
        output_path.write_text(json.dumps({"speakers": speakers, "stale": False}))
        return output_path

    import pipeline.export_ui as export_ui_module

    monkeypatch.setattr(export_ui_module, "export_ui", _export_ui)

    run_pipeline_module.run_pipeline(job_id=job_id, video_path=tmp_path / "video.mp4")

    diarization_segments = json.loads((job_dir / "diarization.json").read_text())["segments"]
    ui_payload = json.loads((job_dir / "ui.json").read_text())

    assert len({seg["speaker_id"] for seg in diarization_segments}) == 2
    assert len({seg["speaker_id"] for seg in captured["speakers"]}) == 2
    assert ui_payload["stale"] is False
