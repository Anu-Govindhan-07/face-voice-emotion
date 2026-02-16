from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.name_tagging.name_extraction import extract_name_signals


_LANGUAGE_HINTS: Dict[str, tuple[str, ...]] = {
    "sv": ("jag", "heter", "mitt", "namn", "det", "här", "och", "är"),
    "en": ("hello", "i'm", "i am", "my name is", "this is", "and"),
    "es": ("hola", "me llamo", "mi nombre", "y", "este es"),
    "fr": ("bonjour", "je m'appelle", "mon nom", "et", "c'est"),
}


def _detect_language(text: str) -> tuple[str, float]:
    lowered = (text or "").casefold()
    if not lowered.strip():
        return "unknown", 0.0
    scores = {
        lang: sum(1 for token in hints if token in lowered)
        for lang, hints in _LANGUAGE_HINTS.items()
    }
    best_lang, best_score = max(scores.items(), key=lambda item: item[1])
    total = sum(scores.values())
    if best_score <= 0:
        return "unknown", 0.0
    confidence = best_score / max(total, 1)
    if confidence < 0.45:
        return "unknown", confidence
    return best_lang, confidence


def _signal_confidence(signal: Dict[str, Any], segment: Dict[str, Any], language: str, language_confidence: float) -> float:
    conf = float(signal.get("confidence", 0.0))
    duration = max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
    if duration < 0.6:
        conf -= 0.1
    if language == "unknown" or language_confidence < 0.5:
        conf -= 0.05
    return max(0.0, min(0.99, conf))


def extract_name_signals_from_segments(
    segments: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    config = config or {}
    min_confidence = float(config.get("min_name_confidence", 0.55))
    default_language = str(config.get("default_language") or "").strip() or None
    base = extract_name_signals(segments)

    output: List[Dict[str, Any]] = []
    for idx, row in enumerate(base):
        segment = segments[idx]
        detected_language, language_confidence = _detect_language(str(segment.get("text", "")))
        language = str(segment.get("language") or default_language or detected_language)
        accepted: List[Dict[str, Any]] = []
        for signal in row.get("signals", []):
            conf = _signal_confidence(signal, segment, language, language_confidence)
            if conf < min_confidence:
                continue
            accepted.append(
                {
                    "name": signal.get("name"),
                    "type": signal.get("type"),
                    "confidence": round(conf, 4),
                    "method": signal.get("method", "pattern"),
                }
            )
        output.append(
            {
                "start": float(segment.get("start", segment.get("start_ts", 0.0))),
                "end": float(segment.get("end", segment.get("end_ts", segment.get("start", 0.0)))),
                "speaker_id": segment.get("speaker_id"),
                "language": language,
                "signals": accepted,
            }
        )
    return output
