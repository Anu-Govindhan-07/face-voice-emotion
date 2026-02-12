from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import FACE_MATCH_MARGIN, FACE_MATCH_THRESHOLD, FACE_MAYBE_THRESHOLD, IDENTITY_STORE
from .utils import console

_STORE_LOCK = threading.Lock()


def _identity_root() -> Path:
    root = IDENTITY_STORE.parent
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    return root


def _load_store() -> Dict:
    _identity_root()
    with _STORE_LOCK:
        if not IDENTITY_STORE.exists():
            return {"identities": []}
        payload = json.loads(IDENTITY_STORE.read_text())
    payload.setdefault("identities", [])
    return payload


def _save_store(store: Dict) -> None:
    _identity_root()
    tmp_path = IDENTITY_STORE.with_suffix(".json.tmp")
    payload = {"identities": store.get("identities", [])}
    with _STORE_LOCK:
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(IDENTITY_STORE)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_vec = np.asarray(a, dtype=np.float32).reshape(-1)
    b_vec = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def _next_identity_id(identities: List[Dict]) -> str:
    highest = 0
    for row in identities:
        raw = str(row.get("id") or "")
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            highest = max(highest, int(digits))
    return f"{highest + 1:04d}"


def _find_identity_by_name(store: Dict, name: str) -> Optional[Dict]:
    for identity in store.get("identities", []):
        if str(identity.get("name", "")).casefold() == str(name or "").casefold():
            return identity
    return None


def _embedding_paths(identity: Dict) -> List[Path]:
    paths = []
    for raw in identity.get("embeddings", []):
        path = Path(str(raw))
        if not path.exists():
            path = _identity_root() / path
        paths.append(path)
    return paths


def match_identity(embedding_path: Path) -> Dict[str, str | float | None]:
    store = _load_store()
    if not embedding_path.exists():
        return {
            "status": "unknown",
            "person_id": None,
            "name": "Unknown",
            "score": 0.0,
            "candidate_person_id": None,
            "candidate_name": "Unknown",
            "candidate_score": 0.0,
            "candidate_margin": 0.0,
        }

    embedding = np.load(embedding_path)
    scores_by_identity: Dict[str, float] = {}
    names_by_identity: Dict[str, str] = {}

    for identity in store.get("identities", []):
        identity_id = str(identity.get("id") or "")
        if not identity_id:
            continue
        names_by_identity[identity_id] = str(identity.get("name") or "Unknown")
        best = -1.0
        for emb_path in _embedding_paths(identity):
            if not emb_path.exists():
                continue
            score = _cosine_similarity(embedding, np.load(emb_path))
            best = max(best, score)
        if best >= 0:
            scores_by_identity[identity_id] = best

    if not scores_by_identity:
        return {
            "status": "unknown",
            "person_id": None,
            "name": "Unknown",
            "score": 0.0,
            "candidate_person_id": None,
            "candidate_name": "Unknown",
            "candidate_score": 0.0,
            "candidate_margin": 0.0,
        }

    ranked = sorted(scores_by_identity.items(), key=lambda item: item[1], reverse=True)
    candidate_id, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = top_score - second_score if second_score >= 0 else 1.0

    status = "unknown"
    resolved_id: Optional[str] = None
    if top_score >= FACE_MATCH_THRESHOLD and margin >= FACE_MATCH_MARGIN:
        status = "matched"
        resolved_id = candidate_id
    elif top_score >= FACE_MAYBE_THRESHOLD:
        status = "maybe"

    resolved_name = names_by_identity.get(resolved_id, "Unknown") if resolved_id else "Unknown"
    candidate_name = names_by_identity.get(candidate_id, "Unknown")

    return {
        "status": status,
        "person_id": resolved_id,
        "name": resolved_name,
        "score": float(top_score),
        "candidate_person_id": candidate_id,
        "candidate_name": candidate_name,
        "candidate_score": float(top_score),
        "candidate_margin": float(max(0.0, margin)),
    }


def remember_identity_embedding(
    track_id: str,
    embedding_path: Path,
    person_id: Optional[str] = None,
    allow_create: bool = False,
) -> Dict[str, Optional[str]]:
    if not embedding_path.exists():
        return {"person_id": None, "name": None}

    store = _load_store()
    identities = store.setdefault("identities", [])
    identity = None

    if person_id:
        identity = next((row for row in identities if str(row.get("id")) == str(person_id)), None)
    if identity is None and allow_create:
        identity = {
            "id": _next_identity_id(identities),
            "name": None,
            "embeddings": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_job": None,
        }
        identities.append(identity)

    if identity is None:
        return {"person_id": None, "name": None}

    emb_dst = _identity_root() / "embeddings" / f"{identity['id']}_{track_id}.npy"
    np.save(emb_dst, np.asarray(np.load(embedding_path), dtype=np.float32))
    emb_ref = str(emb_dst).replace("\\", "/")
    if emb_ref not in identity.setdefault("embeddings", []):
        identity["embeddings"].append(emb_ref)
    _save_store(store)
    return {"person_id": identity["id"], "name": identity.get("name")}


def enroll_identity(
    job_id: str,
    track_id: str,
    name: Optional[str] = None,
    association: Optional[Dict] = None,
    merge_by_name: bool = False,
) -> Dict[str, str]:
    emb_path = Path("data") / "jobs" / job_id / "embeddings" / "face" / f"{track_id}.npy"
    store = _load_store()
    identities = store.setdefault("identities", [])

    identity = None
    if merge_by_name and name:
        identity = _find_identity_by_name(store, name)

    if identity is None and emb_path.exists():
        matched = match_identity(emb_path)
        candidate_id = matched.get("person_id") if matched.get("status") == "matched" else None
        if candidate_id:
            identity = next((row for row in identities if str(row.get("id")) == str(candidate_id)), None)

    if identity is None:
        identity = {
            "id": _next_identity_id(identities),
            "name": name or "Unknown",
            "embeddings": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_job": job_id,
            "associations": [],
        }
        identities.append(identity)

    if name:
        identity["name"] = name

    if association:
        associations = identity.setdefault("associations", [])
        if association not in associations:
            associations.append(association)

    if emb_path.exists():
        emb_dst = _identity_root() / "embeddings" / f"{identity['id']}_{track_id}.npy"
        np.save(emb_dst, np.asarray(np.load(emb_path), dtype=np.float32))
        emb_ref = str(emb_dst).replace("\\", "/")
        if emb_ref not in identity.setdefault("embeddings", []):
            identity["embeddings"].append(emb_ref)

    _save_store(store)
    person_name = str(identity.get("name") or "Unknown")
    console.log(f"Enrolled identity {identity['id']} ({person_name}) for track {track_id}")
    return {"person_id": str(identity["id"]), "name": person_name}


def list_identities() -> List[Dict[str, str | int]]:
    store = _load_store()
    return [
        {
            "person_id": str(identity.get("id") or ""),
            "name": str(identity.get("name") or "Unknown"),
            "count_face_vectors": len(identity.get("embeddings", [])),
        }
        for identity in store.get("identities", [])
    ]
