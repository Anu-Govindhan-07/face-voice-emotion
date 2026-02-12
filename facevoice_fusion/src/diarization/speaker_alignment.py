from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MIN_SEGMENT_DURATION = 0.4


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _merge_short_same_speaker_segments(diarized_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(diarized_segments, key=lambda seg: float(seg.get("start", 0.0)))
    merged: List[Dict[str, Any]] = []
    for seg in ordered:
        speaker = str(seg.get("speaker_id") or "")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if merged:
            prev = merged[-1]
            prev_speaker = str(prev.get("speaker_id") or "")
            prev_end = float(prev.get("end", prev.get("start", 0.0)))
            if speaker and speaker == prev_speaker and (end - start) < _MIN_SEGMENT_DURATION and start <= prev_end + 0.2:
                prev["end"] = max(prev_end, end)
                continue
        merged.append({"speaker_id": speaker, "start": start, "end": end})
    return merged


def align_transcript_to_diarization(
    diarized_segments: List[Dict[str, Any]],
    asr_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    diarized = _merge_short_same_speaker_segments(diarized_segments)
    aligned: List[Dict[str, Any]] = [
        {
            "speaker_id": str(seg.get("speaker_id") or "unknown"),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", seg.get("start", 0.0))),
            "text": "",
            "alignment_confidence": 1.0,
            "unreliable_alignment": False,
        }
        for seg in diarized
    ]

    for asr in asr_segments:
        asr_start = float(asr.get("start", 0.0))
        asr_end = float(asr.get("end", asr_start))
        asr_text = str(asr.get("text") or "")
        words = asr.get("words") or []

        overlap_scores = [
            _overlap(row["start"], row["end"], asr_start, asr_end)
            for row in aligned
        ]
        logger.debug("alignment overlaps asr=[%.2f,%.2f] text=%r scores=%s", asr_start, asr_end, asr_text, overlap_scores)

        if words:
            for word in words:
                w_text = str(word.get("text") or word.get("word") or "").strip()
                if not w_text:
                    continue
                w_start = float(word.get("start", asr_start))
                w_end = float(word.get("end", w_start))
                best_idx = -1
                best_score = 0.0
                for idx, row in enumerate(aligned):
                    score = _overlap(row["start"], row["end"], w_start, w_end)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx >= 0 and best_score > 0:
                    target = aligned[best_idx]
                    target["text"] = (target["text"] + " " + w_text).strip()
                else:
                    best_idx = max(range(len(aligned)), key=lambda i: overlap_scores[i], default=-1)
                    if best_idx >= 0:
                        target = aligned[best_idx]
                        target["text"] = (target["text"] + " " + w_text).strip()
                        target["alignment_confidence"] = min(target["alignment_confidence"], 0.45)
                        target["unreliable_alignment"] = True
        else:
            total = max(0.001, asr_end - asr_start)
            best_idx = max(range(len(aligned)), key=lambda i: overlap_scores[i], default=-1)
            if best_idx < 0:
                continue
            best_overlap = overlap_scores[best_idx]
            confidence = min(1.0, best_overlap / total)
            target = aligned[best_idx]
            target["text"] = (target["text"] + " " + asr_text).strip()
            target["alignment_confidence"] = min(target["alignment_confidence"], confidence)
            if confidence < 0.5:
                target["unreliable_alignment"] = True

    for idx, row in enumerate(aligned):
        overlap_count = 0
        for jdx, other in enumerate(aligned):
            if idx == jdx:
                continue
            if _overlap(row["start"], row["end"], other["start"], other["end"]) > 0:
                overlap_count += 1
        if overlap_count:
            row["alignment_confidence"] = max(0.0, row["alignment_confidence"] - 0.15)

        row["text"] = " ".join(row["text"].split())
        logger.debug(
            "aligned segment speaker=%s [%.2f,%.2f] conf=%.2f unreliable=%s text=%r",
            row["speaker_id"],
            row["start"],
            row["end"],
            row["alignment_confidence"],
            row["unreliable_alignment"],
            row["text"],
        )

    return aligned
