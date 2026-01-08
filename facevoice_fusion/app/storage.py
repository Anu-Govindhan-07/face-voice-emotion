from __future__ import annotations

from pathlib import Path

from .config import JOBS_DIR, UPLOADS_DIR


def ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def job_file(job_id: str, name: str) -> Path:
    return job_dir(job_id) / name


def upload_path(filename: str) -> Path:
    return UPLOADS_DIR / filename
