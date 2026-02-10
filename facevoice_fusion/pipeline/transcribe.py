from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from app.config import ASR_MODEL_NAME
from .utils import console, save_json

_NAME_TOKEN = r"([A-Za-zÅÄÖåäö][A-Za-zÅÄÖåäö\-']{1,30})"
_SELF_IDENTIFICATION_PATTERNS = [
    re.compile(rf"\bmy name is\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bi am\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bi['’]?m\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bjag heter\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bmitt namn är\s+{_NAME_TOKEN}\b", re.IGNORECASE),
]

_MENTION_PATTERNS = [
    re.compile(rf"\b(?:this is|that is|it's|it is|det här är|detta är|där är)\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\b(?:he is|she is|han är|hon är)\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\b(?:called|named|heter)\s+{_NAME_TOKEN}\b", re.IGNORECASE),
]

_STOPWORDS = {
    "jag", "du", "han", "hon", "vi", "ni", "dom", "de", "det", "den", "här", "där",
    "and", "or", "the", "a", "an", "this", "that", "is", "are", "name", "mitt", "namn",
}


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


def _normalize_name(candidate: str) -> Optional[str]:
    cleaned = candidate.strip(" .,!?:;\"'()[]{}")
    if len(cleaned) < 2:
        return None
    lowered = cleaned.casefold()
    if lowered in _STOPWORDS:
        return None
    return cleaned[0].upper() + cleaned[1:].lower()


def _extract_self_identification_name(text: str) -> Optional[str]:
    for pattern in _SELF_IDENTIFICATION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        return _normalize_name(match.group(1))
    return None


def _extract_mentioned_names(text: str) -> List[str]:
    names: List[str] = []
    for pattern in _MENTION_PATTERNS:
        for match in pattern.finditer(text):
            normalized = _normalize_name(match.group(1))
            if normalized:
                names.append(normalized)
    return names


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _best_speaker_for_segment(speakers: List[dict], segment: dict) -> Optional[str]:
    best_speaker_id = None
    best_overlap = 0.0
    seg_start = float(segment.get("start", 0.0))
    seg_end = float(segment.get("end", seg_start))
    for speaker in speakers:
        speaker_id = speaker.get("speaker_id")
        if not speaker_id:
            continue
        overlap = _overlap(float(speaker["start"]), float(speaker["end"]), seg_start, seg_end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker_id = speaker_id
    return best_speaker_id


def infer_name_signals(speakers: List[dict], transcript_segments: List[dict]) -> Dict[str, Dict[str, str]]:
    speaker_self_names: Dict[str, str] = {}
    speaker_mentioned_names: Dict[str, str] = {}
    mention_votes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    latest_segment_by_speaker: Dict[str, dict] = {}

    for segment in transcript_segments:
        segment_text = segment.get("text", "")
        if not segment_text:
            continue
        owner_speaker_id = _best_speaker_for_segment(speakers, segment)
        if not owner_speaker_id:
            continue
        latest_segment_by_speaker[owner_speaker_id] = {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", segment.get("start", 0.0))),
            "mid": (float(segment.get("start", 0.0)) + float(segment.get("end", segment.get("start", 0.0)))) / 2.0,
            "text": segment_text,
        }

        self_name = _extract_self_identification_name(segment_text)
        if self_name:
            speaker_self_names[owner_speaker_id] = self_name

        for mentioned in _extract_mentioned_names(segment_text):
            if mentioned == speaker_self_names.get(owner_speaker_id):
                continue
            mention_votes[mentioned][owner_speaker_id] += 1

    for mentioned_name, votes in mention_votes.items():
        speaker_id = max(votes.items(), key=lambda item: item[1])[0]
        speaker_mentioned_names[speaker_id] = mentioned_name

    return {
        "self": speaker_self_names,
        "mentioned": speaker_mentioned_names,
        "speaker_segments": latest_segment_by_speaker,
    }
