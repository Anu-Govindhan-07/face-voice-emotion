from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

_DEFAULT_NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
_MIN_TEXT_LEN = 2

_SELF_PATTERNS = [
    re.compile(r"\b(?:i am|i['’]m|my name is|jag heter|mitt namn är)\s+([^.!?\n]+)", re.IGNORECASE),
]
_MENTION_PATTERNS = [
    re.compile(r"\b(?:this is|that is|meet|this is my friend|det här är|detta är|träffa|han heter|hon heter)\s+([^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\b([^\W\d_][^\s,.;:!?]{1,30})\s+here\b", re.IGNORECASE),
]
_SELF_CUE = re.compile(r"\b(i am|i['’]m|my name is|jag heter|mitt namn är)\b", re.IGNORECASE)
_MENTION_CUE = re.compile(r"\b(this is|that is|meet|det här är|detta är|träffa|han heter|hon heter|my friend)\b", re.IGNORECASE)

_SPLIT_NAMES = re.compile(r"\s*(?:,|;|&|\band\b|\boch\b|\by\b|\bet\b|\bund\b|\be\b)\s*", re.IGNORECASE)
_NON_NAME_TOKENS = {
    "la",
    "sweden",
    "america",
    "usa",
    "uk",
    "europe",
    "unknown",
    "speaker",
    "and",
    "och",
    "the",
}


class MultilingualNameExtractor:
    def __init__(self, model_name: str = _DEFAULT_NER_MODEL) -> None:
        self.model_name = model_name
        self._ner = None
        self._failed = False

    def _get_pipeline(self):
        if self._ner is not None:
            return self._ner
        if self._failed:
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self._ner = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
        except Exception as exc:
            self._failed = True
            logger.warning("Failed to load multilingual NER model %s: %s", self.model_name, exc)
            return None
        return self._ner

    def extract_person_entities(self, text: str) -> List[Dict[str, Any]]:
        if not text or len(text.strip()) < _MIN_TEXT_LEN:
            return []
        ner = self._get_pipeline()
        if ner is None:
            return []
        try:
            entities = ner(text)
        except Exception as exc:
            logger.warning("NER inference failed: %s", exc)
            return []

        out: List[Dict[str, Any]] = []
        for ent in entities:
            label = str(ent.get("entity_group") or ent.get("entity") or "")
            if "PER" not in label.upper() and "PERSON" not in label.upper():
                continue
            name = _normalize_name(str(ent.get("word") or ""))
            if not name:
                continue
            out.append(
                {
                    "name": name,
                    "start": int(ent.get("start", 0)),
                    "end": int(ent.get("end", 0)),
                    "type": _classify_type_from_context(text, int(ent.get("start", 0))),
                    "method": "ner",
                }
            )
        return out


_EXTRACTOR = MultilingualNameExtractor()


def _normalize_name(raw: str) -> Optional[str]:
    value = unicodedata.normalize("NFKC", (raw or "").strip(" \t\n\r.,!?;:\"'`()[]{}"))
    value = "".join(ch for ch in value if ch.isalpha() or ch in "-'")
    if len(value) < 2:
        return None
    if value.casefold() in _NON_NAME_TOKENS:
        return None
    if value.isupper() and len(value) <= 3:
        return None
    return "-".join(part[:1].upper() + part[1:].lower() for part in value.split("-"))


def _classify_type_from_context(text: str, start: int) -> str:
    context = text[max(0, start - 40) : start]
    if _SELF_CUE.search(context):
        return "self"
    if _MENTION_CUE.search(context):
        return "mentioned"
    return "mentioned"


def _split_names(fragment: str) -> List[str]:
    names: List[str] = []
    for token in _SPLIT_NAMES.split(fragment.strip()):
        if not token:
            continue
        piece = token.split()[0]
        name = _normalize_name(piece)
        if name:
            names.append(name)
    return names


def _extract_pattern_entities(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pattern in _SELF_PATTERNS:
        for match in pattern.finditer(text):
            for name in _split_names(match.group(1)):
                out.append({"name": name, "start": match.start(1), "end": match.end(1), "type": "self", "method": "pattern"})
    for pattern in _MENTION_PATTERNS:
        for match in pattern.finditer(text):
            group = match.group(1)
            for name in _split_names(group):
                out.append({"name": name, "start": match.start(1), "end": match.end(1), "type": "mentioned", "method": "pattern"})
    return out


def _score_signal(method: str, signal_type: str, text: str, start: int, alignment_confidence: float, duration: float) -> float:
    if method == "ner":
        score = 0.75
        context = text[max(0, start - 40) : start]
        if signal_type == "self" and _SELF_CUE.search(context):
            score += 0.15
    else:
        score = 0.85 if signal_type == "self" else 0.70
    if alignment_confidence < 0.5:
        score -= 0.20
    if duration < 0.6:
        score -= 0.10
    return max(0.0, min(0.98, score))


def extract_name_signals(aligned_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for segment in aligned_segments:
        text = str(segment.get("text") or segment.get("transcript_text") or "")
        speaker_id = str(segment.get("speaker_id") or "unknown")
        start = float(segment.get("start", segment.get("start_ts", 0.0)))
        end = float(segment.get("end", segment.get("end_ts", start)))
        alignment_confidence = float(segment.get("alignment_confidence", 1.0))
        duration = max(0.0, end - start)

        ner_entities = _EXTRACTOR.extract_person_entities(text)
        pattern_entities = _extract_pattern_entities(text)
        logger.debug("name_extraction raw_text=%r", text)
        logger.debug("name_extraction ner_entities=%s", ner_entities)
        logger.debug("name_extraction pattern_matches=%s", pattern_entities)

        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in ner_entities + pattern_entities:
            key = (item["name"].casefold(), item["type"])
            score = _score_signal(item["method"], item["type"], text, int(item.get("start", 0)), alignment_confidence, duration)
            current = merged.get(key)
            if current is None or score > current["confidence"]:
                merged[key] = {
                    "name": item["name"],
                    "type": item["type"],
                    "confidence": round(score, 4),
                    "method": item["method"],
                }

        final_signals = sorted(merged.values(), key=lambda row: row["confidence"], reverse=True)
        logger.debug("name_extraction final_signals=%s", final_signals)

        out.append(
            {
                "speaker_id": speaker_id,
                "start": start,
                "end": end,
                "alignment_confidence": round(alignment_confidence, 4),
                "signals": final_signals,
                "text": text,
            }
        )
    return out
