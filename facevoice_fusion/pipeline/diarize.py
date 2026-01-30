from __future__ import annotations

import os
from pathlib import Path
from typing import List

import soundfile as sf

from app.config import HUGGINGFACE_TOKEN
from .utils import console, save_json


def diarize_audio(audio_path: Path, output_path: Path) -> List[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diarization: List[dict] = []

    if HUGGINGFACE_TOKEN:
        try:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
            diar = pipeline(str(audio_path))
            idx = 1
            for turn, _, _ in diar.itertracks(yield_label=True):
                diarization.append({
                    "speaker_id": f"S{idx}",
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "conf": 0.9,
                })
                idx += 1
        except ImportError:
            console.log("Pyannote is not installed; falling back to single-speaker diarization.")
        except Exception as exc:
            console.log(f"Pyannote diarization failed, fallback mode: {exc}")

    if not diarization:
        console.log("Fallback diarization mode")
        duration = sf.info(str(audio_path)).duration
        diarization = [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.2}]

    save_json(output_path, {"segments": diarization})
    return diarization
