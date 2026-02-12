from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

_DEFAULT_NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
_MIN_TEXT_LEN = 3

_SELF_PREFIX_PATTERNS = [
    re.compile(r"\b(?:i am|i['’]m|my name is|jag heter|mitt namn är)\s+([^.!?\n]+)", re.IGNORECASE),
]
_MENTION_PREFIX_PATTERNS = [
    re.compile(r"\b(?:this is|that is|det här är|detta är)\s+([^.!?\n]+)", re.IGNORECASE),
]

_CONJUNCTION_SPLIT = re.compile(r"\s*(?:,|;|\band\b|\boch\b|\by\b|\bet\b|\bund\b|\be\b)\s*", re.IGNORECASE)

_STOPWORDS = {
    "i",
    "im",
    "am",
    "my",
    "name",
    "is",
    "jag",
    "heter",
    "mitt",
    "namn",
    "det",
    "här",
    "detta",
    "this",
    "that",
    "and",
    "och",
}
_NON_NAME_TOKENS = {
    "la",
    "usa",
    "uk",
    "eu",
    "unknown",
    "speaker",
    "american",
    "filipino",
    "born",
    "raised",
    "from",
    "in",
    "a",
    "an",
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

        people: List[Dict[str, Any]] = []
        for ent in entities:
            label = str(ent.get("entity_group") or ent.get("entity") or "")
            if "PER" not in label.upper() and "PERSON" not in label.upper():
                continue
            raw = str(ent.get("word") or "")
            name = _normalize_name(raw)
            if not name:
                continue
            people.append(
                {
                    "name": name,
                    "start": int(ent.get("start", 0)),
                    "end": int(ent.get("end", 0)),
                    "confidence": float(ent.get("score", 0.0)),
                    "source": "ner",
                }
            )
        return people


_EXTRACTOR = MultilingualNameExtractor()


def _normalize_name(raw: str) -> Optional[str]:
    value = unicodedata.normalize("NFKC", (raw or "").strip(" \t\n\r.,!?;:\"'`()[]{}"))
    value = "".join(ch for ch in value if ch.isalpha() or ch in "-'")
    if len(value) < 2:
        return None
    if value.casefold() in _STOPWORDS or value.casefold() in _NON_NAME_TOKENS:
        return None
    if value.isupper() and len(value) <= 3:
        return None
    return "-".join(part[:1].upper() + part[1:].lower() for part in value.split("-"))


def _split_candidate_names(fragment: str) -> List[str]:
    names: List[str] = []
    for part in _CONJUNCTION_SPLIT.split(fragment):
        token = part.strip()
        if not token:
            continue
        first_token = token.split()[0]
        normalized = _normalize_name(first_token)
        if normalized:
            names.append(normalized)
    return names


def _extract_pattern_entities(text: str) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    for pattern in _SELF_PREFIX_PATTERNS:
        for match in pattern.finditer(text):
            span = match.group(1).strip()
            if span.casefold().startswith(("a ", "an ", "the ", "from ", "in ")):
                continue
            for name in _split_candidate_names(span):
                entities.append(
                    {
                        "name": name,
                        "start": match.start(1),
                        "end": match.end(1),
                        "confidence": 0.92,
                        "type": "self",
                        "source": "pattern",
                    }
                )
    for pattern in _MENTION_PREFIX_PATTERNS:
        for match in pattern.finditer(text):
            span = match.group(1).strip()
            if span.casefold().startswith(("a ", "an ", "the ", "from ", "in ")):
                continue
            for name in _split_candidate_names(span):
                entities.append(
                    {
                        "name": name,
                        "start": match.start(1),
                        "end": match.end(1),
                        "confidence": 0.88,
                        "type": "mentioned",
                        "source": "pattern",
                    }
                )
    return entities


def _classify_type_from_context(text: str, start: int) -> str:
    context = text.casefold()[max(0, start - 28) : start]
    if re.search(r"\b(i am|i['’]m|my name is|jag heter|mitt namn är)\b", context):
        return "self"
    if re.search(r"\b(this is|that is|det här är|detta är)\b", context):
        return "mentioned"
    return "mentioned"


def extract_name_signals(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or segment.get("transcript_text") or "")
        speaker_id = str(segment.get("speaker_id") or "")
        start = float(segment.get("start", segment.get("start_ts", 0.0)))
        end = float(segment.get("end", segment.get("end_ts", start)))

        ner_entities = _EXTRACTOR.extract_person_entities(text)
        pattern_entities = _extract_pattern_entities(text)
        logger.debug("name_extraction text=%r ner_entities=%s", text, ner_entities)
        logger.debug("name_extraction text=%r pattern_matches=%s", text, pattern_entities)

        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for ent in ner_entities:
            signal_type = _classify_type_from_context(text, int(ent.get("start", 0)))
            key = (ent["name"].casefold(), signal_type)
            score = float(ent.get("confidence", 0.0))
            prev = merged.get(key)
            if prev is None or score > prev["confidence"]:
                merged[key] = {"name": ent["name"], "type": signal_type, "confidence": score}

        for ent in pattern_entities:
            key = (ent["name"].casefold(), ent["type"])
            score = float(ent["confidence"])
            prev = merged.get(key)
            if prev is None or score > prev["confidence"]:
                merged[key] = {"name": ent["name"], "type": ent["type"], "confidence": score}

        signals = sorted(
            (
                {
                    "name": item["name"],
                    "type": item["type"],
                    "confidence": round(float(item["confidence"]), 4),
                }
                for item in merged.values()
            ),
            key=lambda row: row["confidence"],
            reverse=True,
        )
        logger.debug("name_extraction text=%r final_signals=%s", text, signals)

        out.append(
            {
                "speaker_id": speaker_id,
                "start": start,
                "end": end,
                "signals": signals,
                "text": text,
            }
        )
    return out
