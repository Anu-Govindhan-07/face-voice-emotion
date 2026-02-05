from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from app.config import ASR_MODEL_NAME
from .utils import console, save_json

_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Z][a-z]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bi am\s+([A-Z][a-z]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bi['’]?m\s+([A-Z][a-z]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bthis is\s+([A-Z][a-z]{1,30})\b", re.IGNORECASE),
]


def transcribe_audio(audio_path: Path, output_path: Path) -> List[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments: List[dict] = []

    try:
        from transformers import pipeline

        asr = pipeline(
            task="automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            chunk_length_s=20,
            stride_length_s=4,
            return_timestamps=True,
        )
        result = asr(str(audio_path), return_timestamps=True)
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        for chunk in chunks:
            ts = chunk.get("timestamp")
            if not ts or len(ts) != 2:
                continue
            start, end = ts
            if start is None or end is None:
                continue
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            segments.append({"start": float(start), "end": float(end), "text": text})
    except Exception as exc:
        console.log(f"ASR failed; continuing without transcript: {exc}")

    save_json(output_path, {"segments": segments})
    return segments


def _extract_name(text: str) -> Optional[str]:
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            return candidate[0].upper() + candidate[1:].lower()
    return None


def infer_speaker_names(speakers: List[dict], transcript_segments: List[dict]) -> Dict[str, str]:
    speaker_names: Dict[str, str] = {}
    for speaker in speakers:
        speaker_id = speaker.get("speaker_id")
        if not speaker_id:
            continue
        best_name = None
        best_overlap = 0.0
        for segment in transcript_segments:
            name = _extract_name(segment.get("text", ""))
            if not name:
                continue
            overlap = max(0.0, min(speaker["end"], segment["end"]) - max(speaker["start"], segment["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = name
        if best_name:
            speaker_names[speaker_id] = best_name
    return speaker_names
