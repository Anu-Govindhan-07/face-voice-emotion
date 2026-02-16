from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from pipeline.diarize import _fallback_multi_speaker_diarization


def _tone(freq: float, seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return 0.15 * np.sin(2 * np.pi * freq * t)


def test_fallback_diarization_splits_alternating_speakers(tmp_path):
    sr = 16000
    # Alternate two very different "voices" (synthetic tones) over 6 seconds.
    audio = np.concatenate([
        _tone(180, 1.5, sr),
        _tone(430, 1.5, sr),
        _tone(180, 1.5, sr),
        _tone(430, 1.5, sr),
    ]).astype(np.float32)
    wav_path = tmp_path / "two_speakers.wav"
    sf.write(wav_path, audio, sr)

    diar = _fallback_multi_speaker_diarization(wav_path)
    speakers = {seg["speaker_id"] for seg in diar}

    assert len(diar) >= 2
    assert len(speakers) >= 2
