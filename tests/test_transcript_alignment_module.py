from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.diarization.transcript_alignment import align_asr_to_diarization


def test_align_asr_to_diarization_overlap_mapping():
    diarized = [
        {"speaker_id": "S1", "start": 0.0, "end": 1.0},
        {"speaker_id": "S2", "start": 1.0, "end": 2.0},
    ]
    asr = [{"start": 0.1, "end": 0.9, "text": "hello from one"}, {"start": 1.1, "end": 1.9, "text": "hello from two"}]
    out = align_asr_to_diarization(diarized, asr)
    s1 = next(seg for seg in out if seg["speaker_id"] == "S1")
    s2 = next(seg for seg in out if seg["speaker_id"] == "S2")
    assert "one" in s1["text"]
    assert "two" in s2["text"]
    assert s1["alignment_confidence"] > 0.7
