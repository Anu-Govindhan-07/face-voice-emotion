from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.name_tagging.name_extraction import extract_name_signals


def _signals(text: str, alignment_confidence: float = 1.0):
    rows = extract_name_signals(
        [{"start": 0.0, "end": 1.0, "speaker_id": "S1", "text": text, "alignment_confidence": alignment_confidence}]
    )
    return rows[0]["signals"]


def test_this_is_taylor_detected_as_mentioned():
    signals = _signals("This is Taylor.")
    assert any(s["name"] == "Taylor" and s["type"] == "mentioned" and s["confidence"] >= 0.7 for s in signals)


def test_meet_taylor_detected_as_mentioned():
    signals = _signals("Meet Taylor.")
    assert any(s["name"] == "Taylor" and s["type"] == "mentioned" and s["confidence"] >= 0.7 for s in signals)


def test_im_taylor_detected_as_self():
    signals = _signals("I'm Taylor.")
    assert any(s["name"] == "Taylor" and s["type"] == "self" and s["confidence"] >= 0.8 for s in signals)


def test_swedish_self_intro():
    signals = _signals("Jag heter Anu.")
    assert any(s["name"] == "Anu" and s["type"] == "self" for s in signals)


def test_swedish_mentioned_intro():
    signals = _signals("Det här är Sam.")
    assert any(s["name"] == "Sam" and s["type"] == "mentioned" for s in signals)


def test_multi_name_self_intro():
    signals = _signals("I'm Matthew and Sina.")
    assert any(s["name"] == "Matthew" and s["type"] == "self" and s["confidence"] >= 0.8 for s in signals)
    assert any(s["name"] == "Sina" and s["type"] == "self" and s["confidence"] >= 0.8 for s in signals)


def test_low_alignment_reduces_confidence():
    signals = _signals("I'm Taylor.", alignment_confidence=0.3)
    taylor = next(s for s in signals if s["name"] == "Taylor")
    assert taylor["confidence"] < 0.8


def test_noise_la_not_person():
    signals = _signals("born and raised in LA")
    assert not any(s["name"] == "La" for s in signals)
