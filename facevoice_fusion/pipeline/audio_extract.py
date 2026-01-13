from __future__ import annotations

from pathlib import Path

import ffmpeg

from app.config import AUDIO_SAMPLE_RATE
from .utils import console


def extract_audio(video_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.log(f"Extracting audio from {video_path}")
    (
        ffmpeg
        .input(str(video_path))
        .output(str(output_path), ac=1, ar=AUDIO_SAMPLE_RATE, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path
