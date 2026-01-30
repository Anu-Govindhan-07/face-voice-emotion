from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import List

import ffmpeg
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rich.console import Console

from .config import IDENTITY_STORE, STATIC_DIR
from .events import broadcaster
from .jobs import job_store
from .models import (
    IdentityEnrollRequest,
    IdentityEnrollResponse,
    IdentityListItem,
    JobStatusResponse,
    UploadResponse,
)
from .storage import ensure_dirs, job_dir, job_file
from pipeline.identity import enroll_identity, list_identities
from pipeline.run_pipeline import run_pipeline

console = Console()

app = FastAPI(title="FaceVoice Fusion")
PREVIEW_FILENAME = "input_video_preview.mp4"
PREVIEW_COMPATIBLE_EXTENSIONS = {".mp4", ".webm", ".ogg", ".ogv"}


def _load_ui(job_id: str) -> Path:
    ui_path = job_file(job_id, "ui.json")
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="ui.json not found")
    return ui_path


def _load_video(job_id: str) -> Path:
    job_path = job_dir(job_id)
    preview_path = job_path / PREVIEW_FILENAME
    if preview_path.exists():
        return preview_path
    matches = sorted(job_path.glob("input_video.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="video not found")
    return matches[0]


def _preview_path(job_id: str) -> Path:
    return job_dir(job_id) / PREVIEW_FILENAME


def create_preview_video(video_path: Path, preview_path: Path) -> None:
    if preview_path.exists():
        return
    console.log(f"Creating preview video at {preview_path}")
    try:
        (
            ffmpeg.input(str(video_path))
            .output(
                str(preview_path),
                vcodec="libx264",
                acodec="aac",
                movflags="faststart",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        console.log(f"Failed to create preview video: {exc}")


@app.on_event("startup")
async def on_startup() -> None:
    ensure_dirs()
    IDENTITY_STORE.parent.mkdir(parents=True, exist_ok=True)
    if not IDENTITY_STORE.exists():
        IDENTITY_STORE.write_text(json.dumps({"persons": {}, "meta": {"schema_version": 1}}, indent=2))


@app.post("/upload", response_model=UploadResponse)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> UploadResponse:
    job = job_store.create_job()
    job_path = job_dir(job.job_id)
    job_path.mkdir(parents=True, exist_ok=True)
    video_path = job_path / f"input_video{Path(file.filename or '').suffix}"
    content = await file.read()
    video_path.write_bytes(content)
    console.log(f"Saved upload to {video_path}")
    if video_path.suffix.lower() not in PREVIEW_COMPATIBLE_EXTENSIONS:
        background_tasks.add_task(create_preview_video, video_path, _preview_path(job.job_id))
    background_tasks.add_task(run_pipeline, job.job_id, video_path)
    return UploadResponse(job_id=job.job_id, video_id=job.video_id, status=job.status)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        error=job.error,
        artifacts=job.artifacts,
    )


@app.get("/jobs/{job_id}/ui")
async def get_ui(job_id: str) -> FileResponse:
    ui_path = _load_ui(job_id)
    return FileResponse(ui_path)


@app.get("/jobs/{job_id}/video")
async def get_video(job_id: str) -> FileResponse:
    video_path = _load_video(job_id)
    media_type, _ = mimetypes.guess_type(video_path.name)
    return FileResponse(video_path, media_type=media_type)


@app.get("/jobs/{job_id}/events")
async def get_events(job_id: str) -> StreamingResponse:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(broadcaster.stream(job_id), media_type="text/event-stream")


@app.post("/identity/enroll", response_model=IdentityEnrollResponse)
async def identity_enroll(payload: IdentityEnrollRequest) -> IdentityEnrollResponse:
    result = enroll_identity(payload.job_id, payload.track_id, payload.name)
    return IdentityEnrollResponse(person_id=result["person_id"], name=result["name"])


@app.get("/identity/list", response_model=List[IdentityListItem])
async def identity_list() -> List[IdentityListItem]:
    return [IdentityListItem(**item) for item in list_identities()]


@app.get("/")
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text())


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
