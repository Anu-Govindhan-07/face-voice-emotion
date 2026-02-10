from __future__ import annotations

from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.name_tagging.final_name_assignment import assign_names


class MockIdentityStore:
    def __init__(self, by_track: dict[str, dict]):
        self.by_track = by_track
        self.enroll_calls = []

    def match(self, embedding_path: Path):
        return self.by_track.get(embedding_path.stem, {"status": "unknown", "score": 0.0, "name": "Unknown"})

    def enroll(self, name: str, embedding_path: Path, metadata: dict):
        self.enroll_calls.append((name, embedding_path.stem, metadata))
        return {"status": "ok"}


def _mk_track(track_id: str, start: float, end: float, area: int = 10000):
    return {
        "track_id": track_id,
        "start_ts": start,
        "end_ts": end,
        "bboxes": [
            {"t": start, "w": area**0.5, "h": area**0.5},
            {"t": (start + end) / 2, "w": area**0.5, "h": area**0.5},
            {"t": end, "w": area**0.5, "h": area**0.5},
        ],
    }


def _write_emb(job_id: str, track_id: str):
    emb_dir = Path("data") / "jobs" / job_id / "embeddings" / "face"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / f"{track_id}.npy", np.ones((4,), dtype=np.float32))


def test_self_intro_assignment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "job_self"
    _write_emb(job_id, "t1")
    tracks = [_mk_track("t1", 0, 10)]
    segments = [{"speaker_id": "S1", "start_ts": 1.0, "end_ts": 2.0, "transcript_text": "jag heter anu"}]
    asd = {1.2: {"t1": 0.95}}
    store = MockIdentityStore({})

    result = assign_names(job_id, tracks, segments, store, asd=asd)
    assert result["tracks"][0]["label"] == "Anu"
    assert result["tracks"][0]["label_source"] == "self_intro"
    assert store.enroll_calls, "high-confidence self intro should auto-enroll"


def test_mention_assignment_to_non_speaker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "job_mention"
    _write_emb(job_id, "speaker_track")
    _write_emb(job_id, "target_track")
    tracks = [
        _mk_track("speaker_track", 0, 8, area=9000),
        _mk_track("target_track", 3.5, 9, area=14000),
    ]
    segments = [{"speaker_id": "S1", "start_ts": 4.0, "end_ts": 5.0, "transcript_text": "this is sara"}]
    asd = {4.2: {"speaker_track": 0.9, "target_track": 0.1}}

    result = assign_names(job_id, tracks, segments, MockIdentityStore({}), asd=asd)
    by_track = {item["track_id"]: item for item in result["tracks"]}
    assert by_track["speaker_track"]["label"] == "Unknown"
    assert by_track["target_track"]["label"] == "Sara"
    assert by_track["target_track"]["label_source"] == "mention"


def test_out_of_scene_mention_not_assigned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "job_out_scene"
    _write_emb(job_id, "t1")
    tracks = [_mk_track("t1", 0, 4)]
    segments = [{"speaker_id": "S1", "start_ts": 6.0, "end_ts": 7.0, "transcript_text": "det här är erik"}]

    result = assign_names(job_id, tracks, segments, MockIdentityStore({}))
    assert result["tracks"][0]["label"] == "Unknown"
    mention_events = [e for e in result["event_log"] if e["type"] == "mention"]
    assert mention_events and mention_events[0]["track_id"] is None


def test_identity_store_matched_maybe_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "job_match_states"
    for tid in ("t1", "t2", "t3"):
        _write_emb(job_id, tid)
    tracks = [_mk_track("t1", 0, 2), _mk_track("t2", 0, 2), _mk_track("t3", 0, 2)]
    store = MockIdentityStore(
        {
            "t1": {"status": "matched", "score": 0.9, "name": "Mia"},
            "t2": {"status": "maybe", "score": 0.65, "name": "Noah"},
            "t3": {"status": "unknown", "score": 0.1, "name": "Unknown"},
        }
    )

    result = assign_names(job_id, tracks, [], store)
    by_track = {item["track_id"]: item for item in result["tracks"]}
    assert by_track["t1"]["label"] == "Mia"
    assert by_track["t1"]["label_source"] == "identity_store"
    assert by_track["t2"]["label"] == "Unknown"
    assert by_track["t2"]["metadata"]["candidate"]["name"] == "Noah"
    assert by_track["t3"]["label"] == "Unknown"
