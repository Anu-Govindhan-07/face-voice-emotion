from __future__ import annotations

import re
import requests
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.config import (
    ASR_CHUNK_LENGTH_S,
    ASR_LANGUAGE_HINT,
    ASR_MODEL_NAME,
    OPENAI_API_KEY,
    ASR_NUM_BEAMS,
    ASR_STRIDE_LENGTH_S,
)
from .utils import console, load_json, save_json

# ============================================================
# Name extraction regex (kept from your code)
# ============================================================


def _is_openai_transcribe_model(model_name: str) -> bool:
    return str(model_name or "").startswith("gpt-4o-")


def _is_openai_diarize_model(model_name: str) -> bool:
    return str(model_name or "").strip() == "gpt-4o-transcribe-diarize"


def _normalize_openai_speaker(raw_value: Any, label_map: Dict[str, str]) -> Optional[str]:
    if raw_value is None:
        return None
    val = str(raw_value).strip()
    if not val:
        return None
    digits = "".join(ch for ch in val if ch.isdigit())
    if digits:
        n = int(digits)
        if "speaker" in val.casefold():
            n += 1
        return f"S{n}"
    if val not in label_map:
        label_map[val] = f"S{len(label_map) + 1}"
    return label_map[val]


def _transcribe_audio_openai(audio_path: Path) -> List[dict]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    with audio_path.open("rb") as audio_file:
        files = {"file": (audio_path.name, audio_file, "audio/wav")}
        data = {
            "model": ASR_MODEL_NAME,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        if ASR_LANGUAGE_HINT:
            data["language"] = ASR_LANGUAGE_HINT

        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            data=data,
            files=files,
            timeout=180,
        )

    response.raise_for_status()
    payload = response.json()

    segments: List[dict] = []
    label_map: Dict[str, str] = {}
    raw_chunks = payload.get("segments", []) or payload.get("utterances", []) or []
    for chunk in raw_chunks:
        start = chunk.get("start")
        end = chunk.get("end")
        if start is None or end is None:
            continue
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        segment = {"start": float(start), "end": float(end), "text": text}
        speaker_id = _normalize_openai_speaker(
            chunk.get("speaker") or chunk.get("speaker_id") or chunk.get("speaker_label"),
            label_map,
        )
        if speaker_id:
            segment["speaker_id"] = speaker_id
        segments.append(segment)

    return segments

_NAME_TOKEN = r"([A-Za-zÅÄÖåäö][A-Za-zÅÄÖåäö\-']{1,30}(?:\s+[A-Za-zÅÄÖåäö][A-Za-zÅÄÖåäö\-']{1,30})?)"

_SELF_IDENTIFICATION_PATTERNS = [
    re.compile(rf"\bmy name is\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bi am\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bi['’]?m\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bjag heter\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bmitt namn är\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bich bin\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\bdas bin ich[,\s]+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\b(?:ueber|über) mich[,\s]+{_NAME_TOKEN}\b", re.IGNORECASE),
]

_MENTION_PATTERNS = [
    re.compile(
        rf"\b(?:this is|that is|it's|it is|det här är|detta är|där är|das ist)\s+{_NAME_TOKEN}\b",
        re.IGNORECASE,
    ),
    re.compile(rf"\b(?:he is|she is|han är|hon är|er ist|sie ist)\s+{_NAME_TOKEN}\b", re.IGNORECASE),
    re.compile(rf"\b(?:called|named|heter|heisst|heißt)\s+{_NAME_TOKEN}\b", re.IGNORECASE),
]

_STOPWORDS = {
    "jag", "du", "han", "hon", "vi", "ni", "dom", "de", "det", "den", "här", "där",
    "and", "or", "the", "a", "an", "this", "that", "is", "are", "name", "mitt", "namn",
    "im", "am",
}
_NAME_SPLITTER = re.compile(r"\s+(?:and|och|or|eller|und)\s+", re.IGNORECASE)

# ============================================================
# ASR: Transcribe audio into timestamped segments
# ============================================================

def transcribe_audio(audio_path: Path, output_path: Path) -> List[dict]:
    """
    Runs ASR and outputs segments:
      [{start, end, text}, ...]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments: List[dict] = []

    try:
        if _is_openai_transcribe_model(ASR_MODEL_NAME):
            segments = _transcribe_audio_openai(audio_path)
        else:
            from transformers import pipeline

            asr = pipeline(
                task="automatic-speech-recognition",
                model=ASR_MODEL_NAME,
                chunk_length_s=ASR_CHUNK_LENGTH_S,
                stride_length_s=ASR_STRIDE_LENGTH_S,
                return_timestamps=True,
                model_kwargs={"attn_implementation": "sdpa"},
            )
            generate_kwargs: Dict[str, Any] = {"num_beams": ASR_NUM_BEAMS}
            if ASR_LANGUAGE_HINT:
                generate_kwargs["language"] = ASR_LANGUAGE_HINT

            result = asr(str(audio_path), return_timestamps=True, generate_kwargs=generate_kwargs)

            chunks = result.get("chunks", []) if isinstance(result, dict) else []
            for chunk in chunks:
                ts = chunk.get("timestamp")
                if not ts or len(ts) != 2:
                    continue
                start, end = ts
                if start is None or end is None:
                    continue
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                segments.append({"start": float(start), "end": float(end), "text": text})

    except Exception as exc:
        console.log(f"ASR failed; continuing without transcript: {exc}")

    save_json(output_path, {"segments": segments})
    return segments

# ============================================================
# Name helpers
# ============================================================

def _normalize_name(candidate: str) -> Optional[str]:
    cleaned = candidate.strip(" .,!?:;\"'()[]{}")
    if len(cleaned) < 2:
        return None

    # only keep first name chunk before "and/och/und..."
    cleaned = _NAME_SPLITTER.split(cleaned, maxsplit=1)[0].strip()

    parts = [part for part in cleaned.split() if part]
    if not parts:
        return None

    normalized_parts = []
    for part in parts[:2]:
        lowered = part.casefold()
        if lowered in _STOPWORDS:
            return None
        normalized_parts.append(part[0].upper() + part[1:].lower())

    return " ".join(normalized_parts)


def _extract_self_identification_name(text: str) -> Optional[str]:
    for pattern in _SELF_IDENTIFICATION_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        return _normalize_name(match.group(1))
    return None


def _extract_mentioned_names(text: str) -> List[str]:
    names: List[str] = []
    for pattern in _MENTION_PATTERNS:
        for match in pattern.finditer(text or ""):
            normalized = _normalize_name(match.group(1))
            if normalized:
                names.append(normalized)
    return names


# ============================================================
# Diarization ↔ transcript attribution
# ============================================================

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _best_speaker_for_segment(speakers: List[dict], segment: dict) -> Optional[str]:
    best_speaker_id = None
    best_overlap = 0.0
    seg_start = float(segment.get("start", 0.0))
    seg_end = float(segment.get("end", seg_start))
    seg_mid = (seg_start + seg_end) / 2.0
    nearest_speaker_id = None
    nearest_distance = float("inf")

    for speaker in speakers or []:
        speaker_id = speaker.get("speaker_id")
        if not speaker_id:
            continue
        ov = _overlap(float(speaker.get("start", 0.0)), float(speaker.get("end", 0.0)), seg_start, seg_end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker_id = speaker_id

        # Fallback when there is no direct overlap: choose nearest diarization turn by midpoint.
        speaker_mid = (float(speaker.get("start", 0.0)) + float(speaker.get("end", 0.0))) / 2.0
        distance = abs(seg_mid - speaker_mid)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker_id = speaker_id

    return best_speaker_id or nearest_speaker_id


def attribute_speakers_to_segments(
    speakers: List[dict],
    transcript_segments: List[dict],
    overwrite: bool = False,
) -> List[dict]:
    attributed: List[dict] = []
    for segment in transcript_segments:
        enriched = dict(segment)
        if overwrite:
            enriched.pop("speaker_id", None)
        speaker_id = enriched.get("speaker_id") or _best_speaker_for_segment(speakers, enriched)
        if speaker_id:
            enriched["speaker_id"] = speaker_id
        attributed.append(enriched)
    return attributed


# ============================================================
# Transcript-turn speaker inference (your existing logic)
# ============================================================

_QUESTION_TO_OTHER_PATTERNS = [
    re.compile(r"\bvad heter du\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) your name\b", re.IGNORECASE),
    re.compile(r"\bvad kommer du fr(?:a|å)n\b", re.IGNORECASE),
    re.compile(r"\bwhere are you from\b", re.IGNORECASE),
]


def _asks_other_person(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _QUESTION_TO_OTHER_PATTERNS)


def infer_speakers_from_transcript_turns(transcript_segments: List[dict], max_speakers: int = 6) -> List[dict]:
    """
    Heuristic speaker inference for Q/A introductions.
    Useful when diarization collapses to one speaker.
    """
    if not transcript_segments:
        return []

    speaker_ids = [f"S{idx}" for idx in range(1, max(2, int(max_speakers)) + 1)]
    name_to_speaker: Dict[str, str] = {}
    known_speakers: List[str] = ["S1"]
    current_speaker = "S1"
    switch_next = False

    output: List[dict] = []
    for segment in transcript_segments:
        enriched = dict(segment)
        text = str(enriched.get("text") or "")

        # If the previous segment asked "what's your name?" -> likely next speaker replies
        if switch_next and known_speakers:
            if len(known_speakers) == 1:
                next_speaker = "S2"
                if next_speaker not in known_speakers:
                    known_speakers.append(next_speaker)
            else:
                idx = known_speakers.index(current_speaker) if current_speaker in known_speakers else -1
                next_speaker = known_speakers[(idx + 1) % len(known_speakers)]
            current_speaker = next_speaker
            switch_next = False

        # Detect self-intro name -> bind that name to a speaker id
        intro_name = _extract_self_identification_name(text)
        if intro_name:
            if intro_name not in name_to_speaker:
                if len(name_to_speaker) < len(speaker_ids):
                    sid = speaker_ids[len(name_to_speaker)]
                else:
                    sid = speaker_ids[-1]
                name_to_speaker[intro_name] = sid
                if sid not in known_speakers:
                    known_speakers.append(sid)
            current_speaker = name_to_speaker[intro_name]

        enriched["speaker_id"] = current_speaker
        output.append(enriched)

        # If current segment asks the other person, next segment likely switches speaker
        if _asks_other_person(text):
            switch_next = True

    return output


def build_diarization_from_transcript_segments(transcript_segments: List[dict]) -> List[dict]:
    diarization: List[dict] = []
    ordered = sorted(transcript_segments, key=lambda seg: float(seg.get("start", 0.0)))

    for segment in ordered:
        speaker_id = str(segment.get("speaker_id") or "S1")
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end <= start:
            continue

        # merge contiguous segments from same speaker
        if (
            diarization
            and diarization[-1]["speaker_id"] == speaker_id
            and start <= float(diarization[-1]["end"]) + 0.15
        ):
            diarization[-1]["end"] = max(float(diarization[-1]["end"]), end)
            continue

        diarization.append({"speaker_id": speaker_id, "start": start, "end": end, "conf": 0.5})

    return diarization


# ============================================================
# Robust wrapper: if diarization is only S1, infer turns from transcript
# ============================================================

def _count_unique_speakers(segs: List[dict]) -> int:
    return len({s.get("speaker_id") for s in segs if s.get("speaker_id")})


def _should_refresh_cached_transcript(
    transcript_segments: List[dict],
    diarization_segments: List[dict],
) -> bool:
    """
    Avoid stale single-speaker transcript artifacts in reruns when using OpenAI diarization ASR.
    """
    if not _is_openai_diarize_model(ASR_MODEL_NAME):
        return False
    tr_unique = _count_unique_speakers(transcript_segments or [])
    diar_unique = _count_unique_speakers(diarization_segments or [])
    return bool(transcript_segments) and tr_unique < 2 and diar_unique < 2


def robust_speaker_attribution(
    diarization_segments: List[dict],
    transcript_segments: List[dict],
    max_speakers: int = 6,
) -> Dict[str, List[dict]]:
    """
    If diarization collapses to one speaker, infer speaker turns from transcript patterns
    and rebuild diarization. Returns:
      {
        "diarization": [...],
        "transcript": [...],  # transcript segments with speaker_id
      }
    """
    diar_unique = _count_unique_speakers(diarization_segments or [])
    tr_unique = _count_unique_speakers(transcript_segments or [])
    raw_speakers = sorted({str(seg.get("speaker_id")) for seg in diarization_segments or [] if seg.get("speaker_id")})
    transcript_before = sorted({str(seg.get("speaker_id")) for seg in transcript_segments or [] if seg.get("speaker_id")})
    console.log(f"Raw diarization speakers: {raw_speakers}")
    console.log(f"Transcript speakers before robust attribution: {transcript_before}")

    # Case 1: diarization already has multiple speakers -> force fresh attribution
    if diarization_segments and diar_unique >= 2:
        attributed = attribute_speakers_to_segments(diarization_segments, transcript_segments, overwrite=True)
        transcript_speakers = sorted({str(seg.get("speaker_id")) for seg in attributed if seg.get("speaker_id")})
        console.log(f"Final diarization speakers: {raw_speakers}")
        console.log(f"Transcript speakers after attribution: {transcript_speakers}")
        if len(raw_speakers) < 2 and len(transcript_speakers) >= 2:
            console.log("[yellow]Warning: final diarization is single-speaker while transcript has multiple speakers.[/yellow]")
        return {"diarization": diarization_segments, "transcript": attributed}

    # Case 2.1: cached transcript already has multiple speakers -> preserve + rebuild diarization
    if tr_unique >= 2:
        rebuilt_diar = build_diarization_from_transcript_segments(transcript_segments)
        final_speakers = sorted({str(seg.get("speaker_id")) for seg in rebuilt_diar if seg.get("speaker_id")})
        console.log(f"Final diarization speakers: {final_speakers}")
        console.log(f"Transcript speakers after attribution: {transcript_before}")
        return {"diarization": rebuilt_diar, "transcript": transcript_segments}

    # Case 2.2: diarization single-speaker and transcript single-speaker -> infer speaker turns
    inferred = infer_speakers_from_transcript_turns(transcript_segments, max_speakers=max_speakers)
    inf_unique = _count_unique_speakers(inferred)

    # Case 2.3: inference failed -> keep inputs unchanged (non-destructive)
    if inf_unique < 2:
        final_diar = diarization_segments or []
        final_speakers = sorted({str(seg.get("speaker_id")) for seg in final_diar if seg.get("speaker_id")})
        transcript_speakers = sorted({str(seg.get("speaker_id")) for seg in transcript_segments if seg.get("speaker_id")})
        console.log(f"Final diarization speakers: {final_speakers}")
        console.log(f"Transcript speakers after attribution: {transcript_speakers}")
        if len(final_speakers) < 2 and len(transcript_speakers) >= 2:
            console.log("[yellow]Warning: final diarization is single-speaker while transcript has multiple speakers.[/yellow]")
        return {"diarization": final_diar, "transcript": transcript_segments}

    rebuilt_diar = build_diarization_from_transcript_segments(inferred)
    attributed = attribute_speakers_to_segments(rebuilt_diar, transcript_segments, overwrite=True)
    final_speakers = sorted({str(seg.get("speaker_id")) for seg in rebuilt_diar if seg.get("speaker_id")})
    transcript_speakers = sorted({str(seg.get("speaker_id")) for seg in attributed if seg.get("speaker_id")})
    console.log(f"Final diarization speakers: {final_speakers}")
    console.log(f"Transcript speakers after attribution: {transcript_speakers}")
    if len(final_speakers) < 2 and len(transcript_speakers) >= 2:
        console.log("[yellow]Warning: final diarization is single-speaker while transcript has multiple speakers.[/yellow]")

    return {"diarization": rebuilt_diar, "transcript": attributed}


# ============================================================
# Name signals (kept from your code)
# ============================================================

def infer_name_signals(speakers: List[dict], transcript_segments: List[dict]) -> Dict[str, Dict[str, str]]:
    speaker_self_names: Dict[str, str] = {}
    speaker_mentioned_names: Dict[str, str] = {}
    mention_votes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    latest_segment_by_speaker: Dict[str, dict] = {}

    for segment in transcript_segments:
        segment_text = segment.get("text", "")
        if not segment_text:
            continue

        owner_speaker_id = segment.get("speaker_id") or _best_speaker_for_segment(speakers, segment)
        if not owner_speaker_id:
            continue

        latest_segment_by_speaker[owner_speaker_id] = {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", segment.get("start", 0.0))),
            "mid": (float(segment.get("start", 0.0)) + float(segment.get("end", segment.get("start", 0.0)))) / 2.0,
            "text": segment_text,
        }

        self_name = _extract_self_identification_name(segment_text)
        if self_name:
            speaker_self_names[owner_speaker_id] = self_name

        for mentioned in _extract_mentioned_names(segment_text):
            if mentioned == speaker_self_names.get(owner_speaker_id):
                continue
            mention_votes[mentioned][owner_speaker_id] += 1

    for mentioned_name, votes in mention_votes.items():
        speaker_id = max(votes.items(), key=lambda item: item[1])[0]
        speaker_mentioned_names[speaker_id] = mentioned_name

    return {
        "self": speaker_self_names,
        "mentioned": speaker_mentioned_names,
        "speaker_segments": latest_segment_by_speaker,
    }


# ============================================================
# Convenience: transcribe + attribute speakers robustly
# ============================================================

def transcribe_and_attribute(
    audio_path: Path,
    transcript_output_path: Path,
    diarization_segments: List[dict],
    diarization_output_path: Optional[Path] = None,
    max_speakers: int = 6,
) -> Dict[str, List[dict]]:
    """
    Runs ASR transcription, then attributes speakers.
    If diarization is single-speaker, falls back to transcript-turn inference.
    Optionally writes updated diarization to diarization_output_path.
    Returns:
      {
        "diarization": [...],
        "transcript": [...],
      }
    """
    transcript_segments: List[dict] = []
    if transcript_output_path.exists():
        try:
            transcript_segments = load_json(transcript_output_path).get("segments", [])
        except Exception as exc:
            console.log(f"Failed to read existing transcript artifact; rerunning ASR: {exc}")

    if _should_refresh_cached_transcript(transcript_segments, diarization_segments):
        console.log(
            "Cached transcript appears single-speaker while using gpt-4o-transcribe-diarize; "
            "refreshing transcript from ASR."
        )
        transcript_segments = []

    if not transcript_segments:
        transcript_segments = transcribe_audio(audio_path, transcript_output_path)

    bundle = robust_speaker_attribution(diarization_segments, transcript_segments, max_speakers=max_speakers)

    if diarization_output_path is not None:
        diarization_output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(diarization_output_path, {"segments": bundle["diarization"]})

    # overwrite transcript file with speaker_id-attributed segments
    save_json(transcript_output_path, {"segments": bundle["transcript"]})

    return bundle
