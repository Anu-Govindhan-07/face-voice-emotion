from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .config import JOBS_DIR


@dataclass
class JobState:
    job_id: str
    video_id: str
    status: str = "queued"
    stage: Optional[str] = None
    progress: int = 0
    error: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobState] = {}

    def create_job(self) -> JobState:
        job_id = uuid.uuid4().hex
        video_id = uuid.uuid4().hex
        job = JobState(job_id=job_id, video_id=video_id)
        with self._lock:
            self._jobs[job_id] = job
        (JOBS_DIR / job_id).mkdir(parents=True, exist_ok=True)
        return job

    def get_job(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = progress
            if error is not None:
                job.error = error
            if artifacts is not None:
                job.artifacts.update(artifacts)


job_store = JobStore()
