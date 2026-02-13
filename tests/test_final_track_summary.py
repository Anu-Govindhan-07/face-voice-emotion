from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.pipeline.final_track_summary import build_track_summary


class _StubStore:
    def match(self, _embedding_path: Path):
        return {"status": "maybe", "score": 0.66, "candidate_person_id": "0007", "candidate_name": "Candidate", "person_id": None, "name": "Unknown"}


def test_final_summary_contains_required_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    emb_dir = Path("data/jobs/jobx/embeddings/face")
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "T1.npy", np.zeros((512,), dtype=np.float32))

    tracks = [{"track_id": "T1", "start": 0.0, "end": 3.0, "bboxes": [{"t": 0.5, "w": 50, "h": 50}]}]
    emotions = [{"track_id": "T1", "ts": 1.0, "emotion": "happy", "confidence": 0.88}]
    diarized = [{"speaker_id": "S1", "start": 0.0, "end": 2.0}]
    asr = [{"start": 0.0, "end": 1.0, "text": "Hello, I'm Matthew"}]

    out = build_track_summary("jobx", tracks, emotions, diarized, asr, _StubStore(), config={})
    row = out["tracks"][0]

    assert row["track_id"] == "T1"
    assert row["speaker_id"] == "S1"
    assert isinstance(row["detected_names"], list)
    assert "recognized_identity" in row
    assert "dominant_emotion" in row and "current_emotion" in row
    assert Path("data/jobs/jobx/final_track_summary.json").exists()
