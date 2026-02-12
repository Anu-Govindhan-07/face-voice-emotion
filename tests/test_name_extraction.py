from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.name_tagging.name_extraction import extract_name_signals


def _first_signals(text: str):
    rows = extract_name_signals([{"start": 0.0, "end": 1.0, "speaker_id": "S1", "text": text}])
    return rows[0]["signals"]


def test_self_intro_multiple_names_english():
    signals = _first_signals("Hello, I'm Matthew and Sina. I'm a Filipino American born and raised in LA.")
    assert any(s["name"] == "Matthew" and s["type"] == "self" and s["confidence"] > 0.7 for s in signals)
    assert any(s["name"] == "Sina" and s["type"] == "self" and s["confidence"] > 0.7 for s in signals)


def test_my_name_is_self():
    signals = _first_signals("My name is Anu.")
    assert signals == [{"name": "Anu", "type": "self", "confidence": 0.92}]


def test_swedish_self_intro():
    signals = _first_signals("Jag heter Anu.")
    assert any(s["name"] == "Anu" and s["type"] == "self" for s in signals)


def test_this_is_mention_not_self():
    signals = _first_signals("This is Sam.")
    assert any(s["name"] == "Sam" and s["type"] == "mentioned" for s in signals)
    assert not any(s["name"] == "Sam" and s["type"] == "self" for s in signals)


def test_noise_la_not_person():
    signals = _first_signals("born and raised in LA")
    assert not any(s["name"] == "La" for s in signals)
