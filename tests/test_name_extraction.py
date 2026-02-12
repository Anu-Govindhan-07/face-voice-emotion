from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.name_tagging.name_extraction import extract_name_signals


def _signals(text: str, alignment_confidence: float = 1.0):
    rows = extract_name_signals(
        [{"start": 0.0, "end": 1.0, "speaker_id": "S1", "text": text, "alignment_confidence": alignment_confidence}]
    )
    return rows[0]["signals"], rows[0]["debug"]


def test_im_accountable_to_myself_returns_no_names():
    signals, _ = _signals("I'm accountable to myself")
    assert signals == []


def test_im_taylor_self():
    signals, _ = _signals("I'm Taylor")
    assert any(s["name"] == "Taylor" and s["type"] == "self" for s in signals)


def test_this_is_taylor_mentioned():
    signals, _ = _signals("This is Taylor")
    assert any(s["name"] == "Taylor" and s["type"] == "mentioned" for s in signals)


def test_my_name_is_anu_self():
    signals, _ = _signals("My name is Anu")
    assert any(s["name"] == "Anu" and s["type"] == "self" for s in signals)


def test_hello_im_matthew_and_sina():
    signals, _ = _signals("Hello, I'm Matthew and Sina")
    assert any(s["name"] == "Matthew" and s["type"] == "self" for s in signals)
    assert any(s["name"] == "Sina" and s["type"] == "self" for s in signals)


def test_jag_heter_anu_self():
    signals, _ = _signals("Jag heter Anu")
    assert any(s["name"] == "Anu" and s["type"] == "self" for s in signals)


def test_det_har_ar_sam_mentioned():
    signals, _ = _signals("Det här är Sam")
    assert any(s["name"] == "Sam" and s["type"] == "mentioned" for s in signals)


def test_im_excited_to_be_here_no_name():
    signals, _ = _signals("I'm excited to be here")
    assert signals == []


def test_lowercase_rejected_without_ner_confirmation():
    signals, debug = _signals("anu is here")
    assert signals == []
