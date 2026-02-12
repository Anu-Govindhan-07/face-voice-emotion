import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))


import numpy as np

from pipeline import identity


def test_enroll_and_match_cross_video(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "IDENTITY_STORE", tmp_path / "identity_store" / "identities.json")
    monkeypatch.chdir(tmp_path)

    emb_dir_1 = Path("data/jobs/job_001/embeddings/face")
    emb_dir_1.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir_1 / "track12.npy", np.array([1.0, 0.0], dtype=np.float32))

    emb_dir_2 = Path("data/jobs/job_002/embeddings/face")
    emb_dir_2.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir_2 / "track90.npy", np.array([0.99, 0.01], dtype=np.float32))

    enrolled = identity.enroll_identity("job_001", "track12", name="Xeno", merge_by_name=True)
    assert enrolled["person_id"] == "0001"

    matched = identity.match_identity(emb_dir_2 / "track90.npy")
    assert matched["status"] in {"matched", "maybe"}
    assert matched["candidate_name"] == "Xeno"


def test_merge_by_name_appends_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "IDENTITY_STORE", tmp_path / "identity_store" / "identities.json")
    monkeypatch.chdir(tmp_path)

    for job_id, track_id, vec in [("job_a", "t1", [1.0, 0.0]), ("job_b", "t2", [0.98, 0.02])]:
        emb_dir = Path(f"data/jobs/{job_id}/embeddings/face")
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_dir / f"{track_id}.npy", np.array(vec, dtype=np.float32))
        identity.enroll_identity(job_id, track_id, name="Anu", merge_by_name=True)

    payload = identity._load_store()
    assert len(payload["identities"]) == 1
    assert payload["identities"][0]["name"] == "Anu"
    assert len(payload["identities"][0]["embeddings"]) == 2
