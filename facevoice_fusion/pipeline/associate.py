from __future__ import annotations

from typing import Dict, List

from .utils import save_json


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def associate_speakers(
    tracks: List[dict],
    speakers: List[dict],
    output_path,
    speaker_names: Dict[str, str] | None = None,
) -> List[dict]:
    associations: List[dict] = []
    speaker_names = speaker_names or {}
    for speaker in speakers:
        best_track = None
        best_overlap = 0.0
        for track in tracks:
            overlap = _overlap(track["start"], track["end"], speaker["start"], speaker["end"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_track = track
        if best_track:
            associations.append({
                "speaker_id": speaker["speaker_id"],
                "track_id": best_track["track_id"],
                "overlap_sec": best_overlap,
                "confidence": min(1.0, best_overlap / max(0.01, speaker["end"] - speaker["start"]))
                if best_overlap > 0
                else 0.0,
                "inferred_name": speaker_names.get(speaker["speaker_id"]),
            })
    save_json(output_path, {"associations": associations, "speaker_names": speaker_names})
    return associations
