from __future__ import annotations

from pathlib import Path
from typing import List

import soundfile as sf

from app.config import MODEL_VERSIONS
from .utils import save_json


def export_ui(
    output_path: Path,
    video_path: Path,
    tracks: List[dict],
    speakers: List[dict],
    associations: List[dict],
    artifacts: dict,
) -> Path:
    info = sf.info(str(artifacts["audio_wav"]))
    payload = {
        "video": {
            "video_id": video_path.stem,
            "filename": video_path.name,
            "duration_sec": float(info.duration),
        },
        "tracks": tracks,
        "speakers": speakers,
        "associations": associations,
        "artifacts": artifacts,
        "model_versions": MODEL_VERSIONS,
    }
    save_json(output_path, payload)
    return output_path
