from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from pipeline import transcribe as transcribe_module
from pipeline.transcribe import (
    attribute_speakers_to_segments,
    build_diarization_from_transcript_segments,
    robust_speaker_attribution,
    infer_speakers_from_transcript_turns,
    transcribe_and_attribute,
)


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


def test_transcribe_and_attribute_rebuilds_diarization_when_input_is_single_speaker(tmp_path, monkeypatch):
    transcript_segments = [
        {"start": 0.0, "end": 2.0, "text": "Hej, jag heter Ova. Vad heter du?"},
        {"start": 2.0, "end": 4.0, "text": "Hej, jag heter Tobias."},
        {"start": 4.0, "end": 6.0, "text": "Jag heter Lille Mor."},
    ]

    monkeypatch.setattr(transcribe_module, "transcribe_audio", lambda *_args, **_kwargs: transcript_segments)

    transcript_path = tmp_path / "transcript.json"
    diarization_path = tmp_path / "diarization.json"
    bundle = transcribe_and_attribute(
        audio_path=tmp_path / "audio.wav",
        transcript_output_path=transcript_path,
        diarization_segments=[{"speaker_id": "S1", "start": 0.0, "end": 6.0}],
        diarization_output_path=diarization_path,
        max_speakers=6,
    )

    final_speakers = {seg["speaker_id"] for seg in bundle["diarization"]}
    transcript_payload = json.loads(transcript_path.read_text())
    diarization_payload = json.loads(diarization_path.read_text())

    assert len(final_speakers) >= 2
    assert len({seg["speaker_id"] for seg in transcript_payload["segments"]}) >= 2
    assert len({seg["speaker_id"] for seg in diarization_payload["segments"]}) >= 2


def test_transcribe_and_attribute_uses_existing_transcript_artifact(tmp_path, monkeypatch):
    existing_segments = [
        {"start": 0.0, "end": 2.0, "text": "Hej, jag heter Ova. Vad heter du?"},
        {"start": 2.0, "end": 4.0, "text": "Hej, jag heter Tobias."},
    ]
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps({"segments": existing_segments}))
    diarization_path = tmp_path / "diarization.json"

    def _boom(*_args, **_kwargs):
        raise AssertionError("transcribe_audio should not run when transcript artifact already exists")

    monkeypatch.setattr(transcribe_module, "transcribe_audio", _boom)

    bundle = transcribe_and_attribute(
        audio_path=tmp_path / "audio.wav",
        transcript_output_path=transcript_path,
        diarization_segments=[{"speaker_id": "S1", "start": 0.0, "end": 4.0}],
        diarization_output_path=diarization_path,
        max_speakers=6,
    )

    assert len({seg["speaker_id"] for seg in bundle["diarization"]}) >= 2


def test_attribute_speakers_to_segments_overwrite_reassigns_stale_cached_speaker_ids():
    transcript_segments = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker_id": "S1"},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker_id": "S1"},
    ]
    diarization_segments = [
        {"speaker_id": "S1", "start": 0.0, "end": 1.0},
        {"speaker_id": "S2", "start": 1.0, "end": 2.0},
    ]

    attributed = attribute_speakers_to_segments(diarization_segments, transcript_segments, overwrite=True)

    assert attributed[0]["speaker_id"] == "S1"
    assert attributed[1]["speaker_id"] == "S2"


def test_robust_speaker_attribution_overwrites_stale_transcript_when_diarization_is_multi_speaker():
    transcript_segments = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker_id": "S1"},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker_id": "S1"},
    ]
    diarization_segments = [
        {"speaker_id": "S1", "start": 0.0, "end": 1.0},
        {"speaker_id": "S2", "start": 1.0, "end": 2.0},
    ]

    bundle = robust_speaker_attribution(diarization_segments, transcript_segments)

    assert bundle["transcript"][0]["speaker_id"] == "S1"
    assert bundle["transcript"][1]["speaker_id"] == "S2"


def test_robust_speaker_attribution_uses_nearest_turn_when_no_overlap():
    transcript_segments = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker_id": "S1"},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker_id": "S1"},
    ]
    # Gap between diarization turns and transcript boundaries can happen due to model timing drift.
    diarization_segments = [
        {"speaker_id": "S1", "start": 2.0, "end": 3.0},
        {"speaker_id": "S2", "start": 3.0, "end": 4.0},
    ]

    bundle = robust_speaker_attribution(diarization_segments, transcript_segments)

    assert bundle["transcript"][0]["speaker_id"] == "S1"
    assert bundle["transcript"][1]["speaker_id"] == "S1"


def test_robust_speaker_attribution_preserves_cached_multi_speaker_transcript_when_diarization_single_speaker():
    transcript_segments = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker_id": "S1"},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker_id": "S2"},
    ]
    diarization_segments = [{"speaker_id": "S1", "start": 0.0, "end": 2.0}]

    bundle = robust_speaker_attribution(diarization_segments, transcript_segments)

    assert [seg["speaker_id"] for seg in bundle["transcript"]] == ["S1", "S2"]
    assert len({seg["speaker_id"] for seg in bundle["diarization"]}) == 2


def test_robust_speaker_attribution_inference_failure_is_non_destructive():
    transcript_segments = [
        {"start": 0.0, "end": 1.0, "text": "hello there"},
        {"start": 1.0, "end": 2.0, "text": "general kenobi"},
    ]
    diarization_segments = [{"speaker_id": "S1", "start": 0.0, "end": 2.0}]

    bundle = robust_speaker_attribution(diarization_segments, transcript_segments)

    assert bundle["diarization"] == diarization_segments
    assert bundle["transcript"] == transcript_segments


def test_transcribe_audio_supports_openai_diarize_model(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    transcript_path = tmp_path / "transcript.json"

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "speaker_0"},
                    {"start": 1.0, "end": 2.0, "text": "hi", "speaker": "speaker_1"},
                ]
            }

    monkeypatch.setattr(transcribe_module, "ASR_MODEL_NAME", "gpt-4o-transcribe-diarize")
    monkeypatch.setattr(transcribe_module, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(transcribe_module.requests, "post", lambda *args, **kwargs: _Resp())

    segments = transcribe_module.transcribe_audio(audio_path, transcript_path)

    assert [seg["speaker_id"] for seg in segments] == ["S1", "S2"]


def test_transcribe_audio_openai_requires_api_key(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    transcript_path = tmp_path / "transcript.json"

    monkeypatch.setattr(transcribe_module, "ASR_MODEL_NAME", "gpt-4o-transcribe")
    monkeypatch.setattr(transcribe_module, "OPENAI_API_KEY", "")

    segments = transcribe_module.transcribe_audio(audio_path, transcript_path)

    assert segments == []
