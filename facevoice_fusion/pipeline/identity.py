from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.config import FACE_MATCH_THRESHOLD, FACE_MAYBE_THRESHOLD, IDENTITY_STORE
from .utils import console


def _load_store() -> Dict:
    if not IDENTITY_STORE.exists():
        return {"persons": {}, "meta": {"schema_version": 1}}
    return json.loads(IDENTITY_STORE.read_text())


def _save_store(store: Dict) -> None:
    IDENTITY_STORE.write_text(json.dumps(store, indent=2))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def match_identity(embedding_path: Path) -> Dict[str, str | float | None]:
    store = _load_store()
    embedding = np.load(embedding_path)
    best_score = -1.0
    best_person_id = None
    best_name = "Unknown"

    for person_id, person in store.get("persons", {}).items():
        for vector in person.get("face_vectors", []):
            vec = np.load(vector["path"])
            score = _cosine_similarity(embedding, vec)
            if score > best_score:
                best_score = score
                best_person_id = person_id
                best_name = person.get("name", "Unknown")

    status = "unknown"
    if best_score >= FACE_MATCH_THRESHOLD:
        status = "matched"
    elif best_score >= FACE_MAYBE_THRESHOLD:
        status = "maybe"

    return {
        "status": status,
        "person_id": best_person_id,
        "name": best_name if best_person_id else "Unknown",
        "score": float(best_score if best_score > 0 else 0.0),
    }


def enroll_identity(job_id: str, track_id: str, name: str) -> Dict[str, str]:
    store = _load_store()
    persons = store.setdefault("persons", {})
    person_id = None
    for pid, person in persons.items():
        if person.get("name") == name:
            person_id = pid
            break
    if person_id is None:
        person_id = f"p_{len(persons) + 1:03d}"
        persons[person_id] = {"name": name, "face_vectors": []}

    emb_path = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{track_id}.npy"
    persons[person_id]["face_vectors"].append({
        "path": str(emb_path),
        "created_at": datetime.utcnow().isoformat(),
    })
    _save_store(store)
    console.log(f"Enrolled identity {person_id} ({name}) for track {track_id}")
    return {"person_id": person_id, "name": name}


def list_identities() -> List[Dict[str, str | int]]:
    store = _load_store()
    items = []
    for person_id, person in store.get("persons", {}).items():
        items.append({
            "person_id": person_id,
            "name": person.get("name", "Unknown"),
            "count_face_vectors": len(person.get("face_vectors", [])),
        })
    return items
