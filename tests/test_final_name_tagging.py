from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.name_tagging.final_name_tagging import (
    enroll_identity,
    load_identity_store,
    match_identity,
    detect_names_from_segments,
    parse_names_from_transcript,
    run_final_name_tagging,
)


def _write_emb(job_id: str, track_id: str, vec: np.ndarray | None = None) -> None:
    emb_dir = Path("data") / "jobs" / job_id / "embeddings" / "face"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / f"{track_id}.npy", vec if vec is not None else np.array([1.0, 0.0], dtype=np.float32))


def _mk_track(track_id: str, start: float, end: float, area: int = 10_000):
    side = area**0.5
    return {
        "track_id": track_id,
        "start_ts": start,
        "end_ts": end,
        "bboxes": [
            {"t": start, "w": side, "h": side},
            {"t": (start + end) / 2.0, "w": side, "h": side},
            {"t": end, "w": side, "h": side},
        ],
    }


def test_parse_swedish_self_intro():
    parsed = parse_names_from_transcript("Hej! jag heter Anu.")
    assert parsed["self_names"] == ["Anu"]


def test_parse_swedish_mention():
    parsed = parse_names_from_transcript("det här är Sam")
    assert parsed["mentioned_names"] == ["Sam"]


def test_out_of_scene_mention_no_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "job_out_scene"
    _write_emb(job_id, "speaker")
    tracks = [_mk_track("speaker", 0.0, 2.0)]
    segs = [{"speaker_id": "S1", "start_ts": 8.0, "end_ts": 9.0, "transcript_text": "det här är Sam"}]

    result = run_final_name_tagging(job_id, tracks, segs, config={"MIN_ASSIGN_CONFIDENCE": 0.6})
    assert result["tracks"][0]["label"] == "Unknown"
    mention_events = [evt for evt in result["event_log"] if evt["type"] == "mention"]
    assert mention_events and mention_events[0]["notes"] == "mentioned_out_of_scene"


def test_matching_threshold_states(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store_dir = "identity_store"
    store = load_identity_store(store_dir)
    enroll_identity("Mia", np.array([1.0, 0.0], dtype=np.float32), {"job_id": "j", "track_id": "t", "timestamp": 1, "confidence": 0.9}, store_dir, store)

    matched = match_identity(np.array([0.9, 0.1], dtype=np.float32), store, store_dir, {"MATCH_THRESHOLD": 0.8, "MAYBE_THRESHOLD": 0.4})
    maybe = match_identity(np.array([0.5, 0.5], dtype=np.float32), store, store_dir, {"MATCH_THRESHOLD": 0.8, "MAYBE_THRESHOLD": 0.4})
    unknown = match_identity(np.array([-1.0, 0.0], dtype=np.float32), store, store_dir, {"MATCH_THRESHOLD": 0.8, "MAYBE_THRESHOLD": 0.4})

    assert matched["status"] == "matched"
    assert maybe["status"] == "maybe"
    assert unknown["status"] == "unknown"


def test_enrollment_writes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = enroll_identity(
        "Anu",
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        {"job_id": "job1", "track_id": "track1", "timestamp": 3.2, "confidence": 0.91},
    )

    identities_path = Path("identity_store") / "identities.json"
    assert identities_path.exists()
    payload = json.loads(identities_path.read_text())
    assert payload["identities"]
    emb_path = Path("identity_store") / result["embedding_file"]
    assert emb_path.exists()
    assert np.load(emb_path).shape == (3,)


def test_self_intro_uses_speaker_timeline_hint_when_segment_is_ambiguous(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "job_hint"
    _write_emb(job_id, "speaker_track")
    _write_emb(job_id, "other_track")
    tracks = [
        _mk_track("speaker_track", 0.0, 20.0, area=12000),
        _mk_track("other_track", 0.0, 5.0, area=11800),
    ]
    segs = [
        {"speaker_id": "S1", "start": 0.0, "end": 8.64, "text": "Hello, I'm Matthew and Sina. I'm a Filipino American born and raised in LA."},
        {"speaker_id": "S1", "start": 8.64, "end": 14.4, "text": "I studied graphic design."},
    ]

    result = run_final_name_tagging(job_id, tracks, segs, config={"SPEAKER_HINT_CONFIDENCE": 0.75})
    by_track = {row["track_id"]: row for row in result["tracks"]}
    assert by_track["speaker_track"]["label"] == "Matthew"
    assert by_track["speaker_track"]["label_source"] == "self_intro"


def test_detect_names_from_segments_multilingual_and_dedup():
    segs = [
        {
            "speaker_id": "S1",
            "start_ts": 0.0,
            "end_ts": 2.0,
            "transcript_text": "jag heter anu och det här är Sam. Sam pratar också.",
        }
    ]

    result = detect_names_from_segments(segs)
    assert len(result) == 1
    names = result[0]["detected_names"]
    assert any(row["name"] == "Anu" and row["type"] == "self" for row in names)
    assert any(row["name"] == "Sam" and row["type"] == "mentioned" for row in names)
    assert len([row for row in names if row["name"] == "Sam"]) == 1


def test_detect_names_from_segments_no_names():
    segs = [{"speaker_id": "S2", "start_ts": 1.0, "end_ts": 1.4, "transcript_text": "mmm ok yes"}]
    result = detect_names_from_segments(segs)
    assert result[0]["detected_names"] == []
