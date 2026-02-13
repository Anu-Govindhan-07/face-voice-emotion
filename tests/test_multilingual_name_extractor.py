from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facevoice_fusion"))

from src.name_tagging.multilingual_name_extractor import extract_name_signals_from_segments


def test_language_detection_swedish_and_name_extracts():
    rows = extract_name_signals_from_segments(
        [{"start": 0.0, "end": 1.5, "speaker_id": "S1", "text": "Hej, jag heter Anu"}],
        config={},
    )
    assert rows[0]["language"] == "sv"
    assert any(s["name"] == "Anu" for s in rows[0]["signals"])


def test_multi_name_support_self_intro():
    rows = extract_name_signals_from_segments(
        [{"start": 0.0, "end": 2.0, "speaker_id": "S1", "text": "Hello, I'm Matthew and Sina"}],
        config={},
    )
    names = {s["name"] for s in rows[0]["signals"] if s["type"] == "self"}
    assert {"Matthew", "Sina"}.issubset(names)


def test_false_positive_accountable_filtered():
    rows = extract_name_signals_from_segments(
        [{"start": 0.0, "end": 2.0, "speaker_id": "S1", "text": "I'm accountable to myself"}],
        config={},
    )
    assert rows[0]["signals"] == []
