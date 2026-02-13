import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.validation.workflow_validator import WorkflowValidator, WorkflowValidatorConfig


def test_validator_reports_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.validation.workflow_validator.JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr("src.validation.workflow_validator.IDENTITY_STORE", tmp_path / "identity_store" / "identities.json")

    cfg = WorkflowValidatorConfig(identity_store_path=tmp_path / "identity_store" / "identities.json")
    report = WorkflowValidator(cfg).validate(job_id="job1", video_path=tmp_path / "in.mp4")
    assert report.has_failures
    assert any(row.stage == "Artifact assertions" and not row.passed for row in report.results)


def test_validator_passes_artifact_stage_when_outputs_exist(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "data" / "jobs"
    monkeypatch.setattr("src.validation.workflow_validator.JOBS_DIR", jobs_dir)

    job_dir = jobs_dir / "job2"
    (job_dir / "embeddings" / "face").mkdir(parents=True, exist_ok=True)
    np.save(job_dir / "embeddings" / "face" / "FT1.npy", np.zeros((512,), dtype=np.float32))
    (job_dir / "audio.wav").write_bytes(b"wav")
    (job_dir / "faces_tracks.json").write_text(json.dumps({"tracks": [{"track_id": "FT1", "bboxes": [{"t": 0.0}], "start": 0.0, "end": 1.0}]}))
    (job_dir / "diarization.json").write_text(json.dumps({"segments": [{"speaker_id": "S1", "start": 0.0, "end": 1.0}]}))
    (job_dir / "transcript.json").write_text(json.dumps({"segments": [{"speaker_id": "S1", "start": 0.0, "end": 1.0, "text": "I'm Alex"}]}))
    (job_dir / "emotions.json").write_text(json.dumps({"tracks": {"FT1": {"timeline": [{"start": 0.0, "end": 0.5, "label": "neutral", "conf": 0.8}]}}}))
    (job_dir / "associations.json").write_text(json.dumps({"tracks": [{"track_id": "FT1", "label": "Alex"}], "event_log": [{"type": "self_intro", "name": "Alex", "speaker_id": "S1", "ts": 0.2}]}))
    (job_dir / "ui.json").write_text("{}")

    store = tmp_path / "identity_store" / "identities.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    emb_saved = tmp_path / "identity_store" / "embeddings" / "0001_FT1.npy"
    emb_saved.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_saved, np.zeros((512,), dtype=np.float32))
    store.write_text(json.dumps({"identities": [{"id": "0001", "name": "Alex", "embeddings": [str(emb_saved)]}]}))

    cfg = WorkflowValidatorConfig(identity_store_path=store)
    report = WorkflowValidator(cfg).validate(job_id="job2", video_path=tmp_path / "in.mp4")
    artifact_row = next(row for row in report.results if row.stage == "Artifact assertions")
    assert artifact_row.passed
