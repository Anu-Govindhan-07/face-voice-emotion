from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.diarization.speaker_alignment import align_transcript_to_diarization
from src.name_tagging.name_extraction import extract_name_signals


def test_word_level_alignment_splits_across_speakers():
    diarized = [
        {"speaker_id": "S1", "start": 0.0, "end": 1.0},
        {"speaker_id": "S2", "start": 1.0, "end": 2.0},
    ]
    asr = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Hi I'm Taylor this is Sam",
            "words": [
                {"text": "Hi", "start": 0.0, "end": 0.2},
                {"text": "I'm", "start": 0.2, "end": 0.4},
                {"text": "Taylor", "start": 0.4, "end": 0.8},
                {"text": "this", "start": 1.1, "end": 1.3},
                {"text": "is", "start": 1.3, "end": 1.5},
                {"text": "Sam", "start": 1.5, "end": 1.8},
            ],
        }
    ]

    aligned = align_transcript_to_diarization(diarized, asr)
    s1 = next(seg for seg in aligned if seg["speaker_id"] == "S1")
    s2 = next(seg for seg in aligned if seg["speaker_id"] == "S2")
    assert "Taylor" in s1["text"]
    assert "Sam" in s2["text"]


def test_low_overlap_alignment_flag_and_confidence_affect_names():
    diarized = [{"speaker_id": "S1", "start": 0.0, "end": 0.5}]
    asr = [{"start": 1.0, "end": 2.0, "text": "I'm Taylor"}]

    aligned = align_transcript_to_diarization(diarized, asr)
    assert aligned[0]["unreliable_alignment"] is True
    assert aligned[0]["alignment_confidence"] < 0.5

    extracted = extract_name_signals(aligned)
    taylor = next(sig for sig in extracted[0]["signals"] if sig["name"] == "Taylor")
    assert taylor["confidence"] < 0.8
