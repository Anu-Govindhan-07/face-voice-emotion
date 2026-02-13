from __future__ import annotations

from typing import Any, Dict, List

from .speaker_alignment import align_transcript_to_diarization


def align_asr_to_diarization(
    diarized_segments: List[Dict[str, Any]],
    asr_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Align ASR text onto diarized speaker segments.

    Returns rows with speaker_id/start/end/text/alignment_confidence.
    """
    return align_transcript_to_diarization(diarized_segments, asr_segments)
