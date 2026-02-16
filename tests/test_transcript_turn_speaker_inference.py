from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from pipeline.transcribe import build_diarization_from_transcript_segments, infer_speakers_from_transcript_turns


def test_turn_inference_assigns_multiple_speakers_from_self_intros_and_questions():
    segments = [
        {"start": 0.0, "end": 5.68, "text": "Hej, jag heter Ova. Vad heter du?"},
        {"start": 5.68, "end": 7.84, "text": "Hej, jag heter Tobias."},
        {"start": 8.88, "end": 11.72, "text": "Jag heter Lille Mor. Vad heter du?"},
        {"start": 11.72, "end": 13.96, "text": "Hej, jag heter Siri."},
    ]

    inferred = infer_speakers_from_transcript_turns(segments, max_speakers=6)
    speakers = [row["speaker_id"] for row in inferred]

    assert len(set(speakers)) >= 3
    assert speakers[0] != speakers[1]


def test_build_diarization_from_inferred_segments_merges_contiguous_ranges():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "S1"},
        {"start": 1.0, "end": 2.0, "speaker_id": "S1"},
        {"start": 2.0, "end": 3.0, "speaker_id": "S2"},
    ]
    diar = build_diarization_from_transcript_segments(segments)
    assert len(diar) == 2
    assert diar[0]["speaker_id"] == "S1"
    assert diar[0]["start"] == 0.0 and diar[0]["end"] == 2.0
