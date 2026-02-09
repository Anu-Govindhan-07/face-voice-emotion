from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import FACE_MATCH_THRESHOLD, FACE_MAYBE_THRESHOLD, IDENTITY_STORE
from .utils import console

_embedding_cache: Dict[str, Tuple[float, np.ndarray]] = {}


def _load_store() -> Dict:
    if not IDENTITY_STORE.exists():
        return {"persons": {}, "meta": {"schema_version": 3}}
    store = json.loads(IDENTITY_STORE.read_text())
    store.setdefault("persons", {})
    store.setdefault("meta", {})
    store["meta"]["schema_version"] = max(3, int(store["meta"].get("schema_version", 1)))
    for person in store["persons"].values():
        person.setdefault("associations", [])
    return store


def _save_store(store: Dict) -> None:
    IDENTITY_STORE.write_text(json.dumps(store, indent=2))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding) + 1e-8
    return embedding / norm


def _load_embedding(path: str) -> Optional[np.ndarray]:
    emb_path = Path(path)
    if not emb_path.exists():
        return None
    key = str(emb_path)
    modified = emb_path.stat().st_mtime
    cached = _embedding_cache.get(key)
    if cached and cached[0] == modified:
        return cached[1]
    embedding = _normalize(np.load(emb_path))
    _embedding_cache[key] = (modified, embedding)
    return embedding


def _build_person_index(person: Dict) -> bool:
    vectors = person.get("face_vectors", [])
    embeddings: List[np.ndarray] = []
    valid_vectors = []
    for vector in vectors:
        embedding = _load_embedding(vector.get("path", ""))
        if embedding is None:
            continue
        embeddings.append(embedding)
        valid_vectors.append(vector)
    person["face_vectors"] = valid_vectors
    if not embeddings:
        person.pop("face_index", None)
        return False
    matrix = np.vstack(embeddings)
    centroid = _normalize(np.mean(matrix, axis=0))
    person["face_index"] = {
        "centroid": centroid.tolist(),
        "count": int(matrix.shape[0]),
        "updated_at": datetime.utcnow().isoformat(),
    }
    return True


def _sync_store_indexes(store: Dict) -> None:
    changed = False
    for person in store.get("persons", {}).values():
        if person.get("face_index"):
            continue
        if _build_person_index(person):
            changed = True
    if changed:
        _save_store(store)


def _record_association(person: Dict, association: Optional[Dict]) -> None:
    if not association:
        return
    associations = person.setdefault("associations", [])
    payload = {
        "job_id": association.get("job_id"),
        "track_id": association.get("track_id"),
        "speaker_id": association.get("speaker_id"),
        "name": association.get("name"),
        "source": association.get("source"),
        "created_at": datetime.utcnow().isoformat(),
    }
    signature = (payload["job_id"], payload["track_id"], payload["speaker_id"], payload["name"], payload["source"])
    for existing in associations:
        existing_signature = (
            existing.get("job_id"),
            existing.get("track_id"),
            existing.get("speaker_id"),
            existing.get("name"),
            existing.get("source"),
        )
        if existing_signature == signature:
            return
    associations.append(payload)


def _infer_name_from_associations(person: Dict) -> Optional[str]:
    counts: Dict[str, int] = {}
    for association in person.get("associations", []):
        candidate = association.get("name")
        if not candidate:
            continue
        counts[candidate] = counts.get(candidate, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def match_identity(embedding_path: Path) -> Dict[str, str | float | None]:
    store = _load_store()
    _sync_store_indexes(store)
    embedding = _normalize(np.load(embedding_path))
    best_score = -1.0
    best_person_id = None
    best_name = "Unknown"
    candidate_ids: List[str] = []

    for person_id, person in store.get("persons", {}).items():
        centroid_values = person.get("face_index", {}).get("centroid")
        if not centroid_values:
            continue
        centroid = np.array(centroid_values, dtype=np.float32)
        score = _cosine_similarity(embedding, centroid)
        candidate_ids.append(person_id)
        if score > best_score:
            best_score = score
            best_person_id = person_id
            best_name = person.get("name") or _infer_name_from_associations(person) or "Unknown"

    if candidate_ids:
        # Refine against raw vectors for the strongest centroid candidates only.
        ranked = sorted(
            candidate_ids,
            key=lambda pid: _cosine_similarity(
                embedding,
                np.array(store["persons"][pid].get("face_index", {}).get("centroid", []), dtype=np.float32),
            ),
            reverse=True,
        )[:3]
        for person_id in ranked:
            person = store["persons"][person_id]
            for vector in person.get("face_vectors", []):
                vec = _load_embedding(vector.get("path", ""))
                if vec is None:
                    continue
                score = _cosine_similarity(embedding, vec)
                if score > best_score:
                    best_score = score
                    best_person_id = person_id
                    best_name = person.get("name") or _infer_name_from_associations(person) or "Unknown"

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


def remember_identity_embedding(
    track_id: str,
    embedding_path: Path,
    person_id: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    store = _load_store()
    persons = store.setdefault("persons", {})
    now = datetime.utcnow().isoformat()
    selected_person_id = person_id
    if selected_person_id is None:
        selected_person_id = f"p_{len(persons) + 1:03d}"
        persons[selected_person_id] = {"name": None, "face_vectors": [], "associations": []}

    person = persons.setdefault(selected_person_id, {"name": None, "face_vectors": [], "associations": []})
    vectors = person.setdefault("face_vectors", [])
    if not any(vector.get("path") == str(embedding_path) for vector in vectors):
        vectors.append({"path": str(embedding_path), "created_at": now, "track_id": track_id})

    _build_person_index(person)
    _save_store(store)
    return {"person_id": selected_person_id, "name": person.get("name")}


def enroll_identity(
    job_id: str,
    track_id: str,
    name: Optional[str] = None,
    association: Optional[Dict] = None,
) -> Dict[str, str]:
    store = _load_store()
    persons = store.setdefault("persons", {})
    emb_path = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{track_id}.npy"

    matched_person_id = None
    if emb_path.exists():
        matched = match_identity(emb_path)
        if matched.get("status") == "matched":
            matched_person_id = matched.get("person_id")

    person_id = matched_person_id
    if person_id is None and name:
        for pid, person in persons.items():
            if person.get("name") == name:
                person_id = pid
                break
    if person_id is None:
        person_id = f"p_{len(persons) + 1:03d}"
        persons[person_id] = {"name": None, "face_vectors": [], "associations": []}

    if name:
        persons[person_id]["name"] = name

    _record_association(
        persons[person_id],
        association
        or {
            "job_id": job_id,
            "track_id": track_id,
            "speaker_id": None,
            "name": name,
            "source": "manual_enroll" if name else "auto_enroll",
        },
    )

    if not persons[person_id].get("name"):
        persons[person_id]["name"] = _infer_name_from_associations(persons[person_id])

    face_vectors = persons[person_id].setdefault("face_vectors", [])
    if not any(vector.get("path") == str(emb_path) for vector in face_vectors):
        face_vectors.append({
            "path": str(emb_path),
            "created_at": datetime.utcnow().isoformat(),
            "track_id": track_id,
        })
    _build_person_index(persons[person_id])
    _save_store(store)
    person_name = persons[person_id].get("name") or "Unknown"
    console.log(f"Enrolled identity {person_id} ({person_name}) for track {track_id}")
    return {"person_id": person_id, "name": person_name}


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
