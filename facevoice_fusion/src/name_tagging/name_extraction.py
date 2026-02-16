from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

_DEFAULT_NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
_NER_MIN_CONF = 0.60
_MIN_TEXT_LEN = 2

_SELF_PATTERNS = [
    re.compile(r"\b(?:i am|i['’]m|my name is|jag heter|mitt namn är)\s+([^.!?\n]+)", re.IGNORECASE),
]
_MENTION_PATTERNS = [
    re.compile(r"\b(?:this is|det här är|meet)\s+([^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\b(?:han heter|hon heter|él se llama|ella se llama|il s'appelle|elle s'appelle)\s+([^.!?\n]+)", re.IGNORECASE),
]

_SELF_CUE = re.compile(r"\b(i am|i['’]m|my name is|jag heter|mitt namn är)\b", re.IGNORECASE)
_MENTION_CUE = re.compile(r"\b(this is|det här är|meet|han heter|hon heter|él se llama|ella se llama|il s'appelle|elle s'appelle)\b", re.IGNORECASE)
_SPLIT_NAMES = re.compile(r"\s*(?:,|;|&|\band\b|\boch\b)\s*", re.IGNORECASE)

_COMMON_WORD_BLOCKLIST = {
    "accountable",
    "responsible",
    "available",
    "ready",
    "okay",
    "alright",
    "excited",
    "happy",
    "grateful",
    "blessed",
    "strong",
    "great",
}
_PRONOUNS_AND_DETERMINERS = {
    "i",
    "me",
    "my",
    "you",
    "we",
    "this",
    "that",
    "he",
    "she",
    "they",
    "it",
    "a",
    "an",
    "the",
}
_LOCATION_OR_ACRONYM = {"la", "us", "uk", "eu", "usa", "sweden", "america"}
_LOWERCASE_NAME_WHITELIST: set[str] = set()


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
            conf = float(ent.get("score", 0.0))
            if conf < _NER_MIN_CONF:
                continue
            raw = str(ent.get("word") or "")
            name = _normalize_name(raw)
            if not name:
                continue
            out.append(
                {
                    "name": name,
                    "start": int(ent.get("start", 0)),
                    "end": int(ent.get("end", 0)),
                    "type": _classify_type_from_context(text, int(ent.get("start", 0))),
                    "method": "ner",
                    "ner_confidence": conf,
                }
            )
        return out


_EXTRACTOR = MultilingualNameExtractor()


def _normalize_name(raw: str) -> Optional[str]:
    value = unicodedata.normalize("NFKC", (raw or "").strip(" \t\n\r.,!?;:\"'`()[]{}"))
    value = " ".join(value.split())
    if not value:
        return None
    cleaned_tokens = []
    for token in value.split():
        normalized = "".join(ch for ch in token if ch.isalpha() or ch in "-'")
        if normalized:
            cleaned_tokens.append(normalized)
    if not cleaned_tokens:
        return None
    return " ".join("-".join(part[:1].upper() + part[1:].lower() for part in tok.split("-")) for tok in cleaned_tokens)


def _looks_title_cased(candidate: str) -> bool:
    tokens = [t for t in candidate.split() if t]
    return bool(tokens) and all(tok[:1].isupper() for tok in tokens)


def is_valid_name_candidate(token_or_span: str, context: str, ner_confirmed: bool = False, allow_lowercase: bool = False) -> bool:
    candidate = (token_or_span or "").strip()
    if not candidate:
        return False
    normalized = _normalize_name(candidate)
    if not normalized:
        return False

    lower = normalized.casefold()
    if len(normalized) < 2:
        return False
    if lower in _PRONOUNS_AND_DETERMINERS:
        return False
    if lower in _COMMON_WORD_BLOCKLIST:
        return False
    if lower in _LOCATION_OR_ACRONYM:
        return False
    if normalized.isupper() and len(normalized) <= 3:
        return False

    # reject lowercase words unless explicitly whitelisted or NER-confirmed
    raw_first = candidate.split()[0]
    if raw_first[:1].islower() and lower not in _LOWERCASE_NAME_WHITELIST and not ner_confirmed and not allow_lowercase:
        return False

    context_l = context.casefold()
    if re.search(r"\bi(?:\s+am|['’]m)\s+\w+\s+(?:to|for|with)\b", context_l):
        return False
    if re.search(r"\bi(?:\s+am|['’]m)\s+\w+\s+to\s+myself\b", context_l):
        return False

    if ner_confirmed:
        return True
    return _looks_title_cased(normalized)


def _classify_type_from_context(text: str, start: int) -> str:
    context = text[max(0, start - 48) : start]
    if _SELF_CUE.search(context):
        return "self"
    if _MENTION_CUE.search(context):
        return "mentioned"
    return "mentioned"


def _split_names(fragment: str) -> List[str]:
    parts = [p.strip() for p in _SPLIT_NAMES.split(fragment.strip()) if p.strip()]
    names: List[str] = []
    for part in parts:
        # keep up to 2 words for names like "Mary Jane"
        tokens = part.split()
        span = " ".join(tokens[:2])
        names.append(span)
    return names


def _extract_pattern_entities(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    hits: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []

    for pattern, typ in [*[(p, "self") for p in _SELF_PATTERNS], *[(p, "mentioned") for p in _MENTION_PATTERNS]]:
        for match in pattern.finditer(text):
            span = match.group(1)
            for candidate in _split_names(span):
                hit = {"candidate": candidate, "type": typ, "span": span}
                hits.append(hit)
                # stricter self-intro rule for "I'm ..."
                if typ == "self" and re.match(r"\b(?:i am|i['’]m)\b", match.group(0), flags=re.IGNORECASE):
                    if not _looks_title_cased(_normalize_name(candidate) or ""):
                        filtered.append({**hit, "reason": "self_intro_not_title_case"})
                        continue
                strong_intro = bool(re.search(r"\b(my name is|jag heter|mitt namn är|this is|det här är|meet)\b", match.group(0), re.IGNORECASE))
                if not is_valid_name_candidate(candidate, match.group(0), ner_confirmed=False, allow_lowercase=strong_intro):
                    filtered.append({**hit, "reason": "failed_validation"})
                    continue
                normalized = _normalize_name(candidate)
                if not normalized:
                    filtered.append({**hit, "reason": "normalize_failed"})
                    continue
                accepted.append(
                    {
                        "name": normalized,
                        "start": match.start(1),
                        "end": match.end(1),
                        "type": typ,
                        "method": "pattern",
                    }
                )
    return accepted, hits, filtered


def _score_signal(method: str, signal_type: str, text: str, start: int, alignment_confidence: float, duration: float) -> float:
    if method == "ner":
        score = 0.75
        context = text[max(0, start - 48) : start]
        if signal_type == "self" and _SELF_CUE.search(context):
            score += 0.15
    else:
        score = 0.85 if signal_type == "self" else 0.70
    if alignment_confidence < 0.5:
        score -= 0.20
    if duration < 0.6:
        score -= 0.10
    return max(0.0, min(0.98, score))


def extract_name_signals(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or segment.get("transcript_text") or "")
        speaker_id = str(segment.get("speaker_id") or "unknown")
        start = float(segment.get("start", segment.get("start_ts", 0.0)))
        end = float(segment.get("end", segment.get("end_ts", start)))
        alignment_confidence = float(segment.get("alignment_confidence", 1.0))
        duration = max(0.0, end - start)

        ner_entities = _EXTRACTOR.extract_person_entities(text)
        validated_ner = [
            item
            for item in ner_entities
            if is_valid_name_candidate(item["name"], text, ner_confirmed=True)
        ]

        pattern_entities: List[Dict[str, Any]] = []
        pattern_hits: List[Dict[str, Any]] = []
        filtered_candidates: List[Dict[str, Any]] = []

        # NER-first: only fallback if no acceptable NER entities
        if not validated_ner:
            pattern_entities, pattern_hits, filtered_candidates = _extract_pattern_entities(text)

        logger.debug("name_extraction raw_text=%r", text)
        logger.debug("name_extraction ner_entities=%s", validated_ner)
        logger.debug("name_extraction pattern_hits=%s", pattern_hits)

        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in validated_ner + pattern_entities:
            if not is_valid_name_candidate(item["name"], text, ner_confirmed=(item["method"] == "ner")):
                filtered_candidates.append({"candidate": item["name"], "reason": "post_merge_validation_failed", "method": item["method"]})
                continue
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
                "debug": {
                    "ner_entities": validated_ner,
                    "pattern_hits": pattern_hits,
                    "filtered_candidates": filtered_candidates,
                },
            }
        )
    return out
