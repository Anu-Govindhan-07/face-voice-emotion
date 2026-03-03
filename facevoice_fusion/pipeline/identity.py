from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "app").exists() or (parent / "pipeline").exists() or (parent / ".git").exists():
            return parent
    return here.parent


def _identity_root() -> Path:
    env_path = os.getenv("FACEVOICE_IDENTITY_STORE")
    root = Path(env_path) if env_path else (_project_root() / "identity_store")
    root.mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    return root


def _identities_json_path() -> Path:
    return _identity_root() / "identities.json"


def _load_store() -> Dict[str, Any]:
    path = _identities_json_path()
    if not path.exists():
        return {"version": 1, "identities": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 1, "identities": []}
        data.setdefault("version", 1)
        data.setdefault("identities", [])
        return data
    except Exception:
        return {"version": 1, "identities": []}


def _save_store(data: Dict[str, Any]) -> None:
    path = _identities_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_identities() -> List[Dict[str, Any]]:
    return _load_store().get("identities", [])


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        return vec
    return vec / n


def _load_embedding(path: Path) -> Optional[np.ndarray]:
    try:
        v = np.load(str(path))
        if v.ndim > 1:
            v = v.reshape(-1)
        return _normalize(v.astype(np.float32))
    except Exception:
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return -1.0
    return float(np.dot(a, b))


def _person_dir(person_id: str) -> Path:
    d = _identity_root() / "embeddings" / person_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _copy_embedding(person_id: str, embedding_path: Path, tag: str) -> Optional[str]:
    src = Path(embedding_path)
    if not src.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dst = _person_dir(person_id) / f"{ts}_{tag}.npy"
    shutil.copy2(src, dst)
    return str(dst)


def _compute_centroid(embedding_paths: List[str]) -> Optional[np.ndarray]:
    vecs = []
    for p in embedding_paths:
        v = _load_embedding(Path(p))
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    c = np.mean(np.stack(vecs, axis=0), axis=0)
    return _normalize(c)


def _rebuild_person(person: Dict[str, Any]) -> None:
    paths = [str(p) for p in person.get("embedding_paths", []) if p]
    person["sample_count"] = len(paths)

    centroid = _compute_centroid(paths)
    if centroid is not None:
        centroid_path = _person_dir(person["person_id"]) / "centroid.npy"
        np.save(str(centroid_path), centroid)
        person["centroid_path"] = str(centroid_path)

    person["updated_at"] = _utc_now()


def _find_person(store: Dict[str, Any], person_id: str) -> Optional[Dict[str, Any]]:
    for p in store.get("identities", []):
        if str(p.get("person_id")) == str(person_id):
            return p
    return None


def _find_person_by_name(store: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    n = str(name or "").strip().casefold()
    if not n:
        return None
    for p in store.get("identities", []):
        if str(p.get("name") or "").strip().casefold() == n:
            return p
    return None


def _create_person(name: str) -> Dict[str, Any]:
    now = _utc_now()
    return {
        "index_no": int(uuid.uuid4().int % 900000 + 100000),
        "person_id": f"P{uuid.uuid4().hex[:12].upper()}",
        "name": str(name).strip(),
        "aliases": [],
        "created_at": now,
        "updated_at": now,
        "sample_count": 0,
        "embedding_paths": [],
        "centroid_path": None,
        "job_ids": [],
        "tracks": [],
        "associations": [],
    }


def _sample_scores(query: np.ndarray, embedding_paths: List[str]) -> List[float]:
    scores: List[float] = []
    for p in embedding_paths:
        v = _load_embedding(Path(p))
        if v is None:
            continue
        scores.append(_cosine(query, v))
    scores.sort(reverse=True)
    return scores


def match_identity(
    embedding_path: Path,
    match_threshold: Optional[float] = None,
    candidate_threshold: Optional[float] = None,
    min_margin: Optional[float] = None,
    exclude_person_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Robust match:
    - use centroid + per-sample agreement
    - stricter rules for identities with only 1-2 samples
    - avoids false positive re-ID drift
    """
    match_threshold = _safe_float(match_threshold, _safe_float(os.getenv("IDENTITY_MATCH_THRESHOLD"), 0.86))
    candidate_threshold = _safe_float(candidate_threshold, _safe_float(os.getenv("IDENTITY_CANDIDATE_THRESHOLD"), 0.80))
    min_margin = _safe_float(min_margin, _safe_float(os.getenv("IDENTITY_MIN_MARGIN"), 0.05))

    query = _load_embedding(Path(embedding_path))
    if query is None:
        return {"status": "unknown", "score": 0.0, "person_id": None, "name": "Unknown", "margin": 0.0}

    exclude = {str(x) for x in (exclude_person_ids or [])}
    store = _load_store()

    scored: List[Dict[str, Any]] = []

    for person in store.get("identities", []):
        pid = str(person.get("person_id"))
        if pid in exclude:
            continue

        paths = [str(p) for p in person.get("embedding_paths", []) if p]
        sample_count = len(paths)
        if sample_count == 0:
            continue

        centroid_path = person.get("centroid_path")
        centroid = _load_embedding(Path(centroid_path)) if centroid_path else None
        if centroid is None:
            centroid = _compute_centroid(paths)
        if centroid is None:
            continue

        centroid_score = _cosine(query, centroid)
        per_sample = _sample_scores(query, paths)
        if not per_sample:
            continue

        top1 = per_sample[0]
        top2 = per_sample[1] if len(per_sample) > 1 else per_sample[0]
        top3 = per_sample[:3]
        median_topk = float(np.median(np.asarray(top3, dtype=np.float32)))
        votes_085 = sum(1 for s in per_sample if s >= 0.85)
        votes_082 = sum(1 for s in per_sample if s >= 0.82)

        # conservative combined score
        combined = 0.45 * centroid_score + 0.35 * top1 + 0.20 * median_topk

        scored.append(
            {
                "person_id": pid,
                "name": person.get("name", "Unknown"),
                "score": float(combined),
                "centroid_score": float(centroid_score),
                "top1_score": float(top1),
                "top2_score": float(top2),
                "median_topk_score": float(median_topk),
                "votes_085": int(votes_085),
                "votes_082": int(votes_082),
                "sample_count": int(sample_count),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    if not scored:
        return {"status": "unknown", "score": 0.0, "person_id": None, "name": "Unknown", "margin": 0.0}

    best = scored[0]
    second = scored[1]["score"] if len(scored) > 1 else -1.0
    margin = float(best["score"] - float(second))

    sample_count = int(best["sample_count"])

    # stricter if we only have 1-2 stored samples
    if sample_count <= 2:
        strong_match = (
            best["centroid_score"] >= 0.90
            and best["top1_score"] >= 0.92
            and margin >= max(min_margin, 0.06)
        )
        weak_candidate = (
            best["centroid_score"] >= 0.84
            and best["top1_score"] >= 0.86
        )
    else:
        strong_match = (
            best["score"] >= match_threshold
            and best["centroid_score"] >= 0.84
            and best["top1_score"] >= 0.88
            and best["median_topk_score"] >= 0.83
            and best["votes_082"] >= 2
            and margin >= min_margin
        )
        weak_candidate = (
            best["score"] >= candidate_threshold
            and best["top1_score"] >= 0.84
        )

    if strong_match:
        return {
            "status": "matched",
            "score": float(best["score"]),
            "margin": margin,
            "person_id": best["person_id"],
            "name": best["name"],
            "top_candidates": scored[:3],
        }

    if weak_candidate:
        return {
            "status": "candidate",
            "score": float(best["score"]),
            "margin": margin,
            "candidate_person_id": best["person_id"],
            "candidate_name": best["name"],
            "person_id": None,
            "name": "Unknown",
            "top_candidates": scored[:3],
        }

    return {
        "status": "unknown",
        "score": float(best["score"]),
        "margin": margin,
        "person_id": None,
        "name": "Unknown",
        "top_candidates": scored[:3],
    }


def enroll_identity(
    job_id: str,
    track_id: str,
    name: str,
    association: Optional[Dict[str, Any]] = None,
    merge_by_name: bool = True,
    embedding_path: Optional[Path] = None,
) -> Dict[str, Any]:
    store = _load_store()
    clean = str(name or "").strip()
    if not clean:
        return {"status": "error", "reason": "empty_name"}

    person = _find_person_by_name(store, clean) if merge_by_name else None
    created = False
    if person is None:
        person = _create_person(clean)
        store["identities"].append(person)
        created = True

    saved = None
    if embedding_path is not None and Path(embedding_path).exists():
        saved = _copy_embedding(person["person_id"], Path(embedding_path), f"{job_id}_{track_id}")
        if saved and saved not in person["embedding_paths"]:
            person["embedding_paths"].append(saved)

    if job_id and job_id not in person["job_ids"]:
        person["job_ids"].append(job_id)
    if track_id and track_id not in person["tracks"]:
        person["tracks"].append(track_id)

    assoc = {
        "job_id": job_id,
        "track_id": track_id,
        "name": clean,
        "source": (association or {}).get("source", "transcript"),
        "speaker_id": (association or {}).get("speaker_id"),
        "created_at": _utc_now(),
        "embedding_path": saved,
        "first_seen_ts": (association or {}).get("first_seen_ts"),
    }
    person.setdefault("associations", []).append(assoc)

    _rebuild_person(person)
    _save_store(store)

    return {
        "status": "created" if created else "updated",
        "person_id": person["person_id"],
        "name": person["name"],
        "embedding_path": saved,
        "sample_count": person.get("sample_count", 0),
    }


def remember_identity_embedding(
    track_id: str,
    embedding_path: Path,
    person_id: str,
    allow_create: bool = False,
) -> Dict[str, Any]:
    """
    Keep this function for explicit/manual memory growth,
    but DO NOT call it automatically from passive re-ID matches.
    """
    store = _load_store()
    person = _find_person(store, str(person_id))
    if person is None:
        if not allow_create:
            return {"status": "missing", "person_id": person_id}
        person = _create_person(name=f"Person-{person_id}")
        person["person_id"] = str(person_id)
        store["identities"].append(person)

    saved = _copy_embedding(str(person_id), Path(embedding_path), f"manual_{track_id}")
    if saved and saved not in person["embedding_paths"]:
        person["embedding_paths"].append(saved)

    _rebuild_person(person)
    _save_store(store)

    return {
        "status": "added",
        "person_id": str(person_id),
        "sample_count": int(person.get("sample_count", 0)),
        "embedding_path": saved,
    }