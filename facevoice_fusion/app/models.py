from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    video_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: Optional[str]
    progress: int
    error: Optional[str]
    artifacts: Dict[str, str]


class IdentityEnrollRequest(BaseModel):
    job_id: str
    track_id: str
    name: Optional[str] = None


class IdentityEnrollResponse(BaseModel):
    person_id: str
    name: str


class IdentityListItem(BaseModel):
    person_id: str
    name: str
    count_face_vectors: int


class BoundingBox(BaseModel):
    t: float
    x: int
    y: int
    w: int
    h: int
    conf: float


class IdentityInfo(BaseModel):
    status: str
    person_id: Optional[str]
    name: str
    score: float


class EmotionSegment(BaseModel):
    start: float
    end: float
    label: str
    conf: float


class EmotionInfo(BaseModel):
    dominant: str
    timeline: List[EmotionSegment]


class TrackInfo(BaseModel):
    track_id: str
    start: float
    end: float
    bboxes: List[BoundingBox]
    identity: IdentityInfo
    emotion: EmotionInfo


class SpeakerSegment(BaseModel):
    speaker_id: str
    start: float
    end: float
    conf: float


class Association(BaseModel):
    speaker_id: str
    track_id: str
    overlap_sec: float
    confidence: float


class Artifacts(BaseModel):
    audio_wav: str
    faces_tracks: str
    diarization: str
    emotions: str
    associations: str


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    duration_sec: float


class ModelVersions(BaseModel):
    face_detector: str
    face_embedder: str
    emotion_model: str
    diarization_model: str


class UIExport(BaseModel):
    video: VideoInfo
    tracks: List[TrackInfo]
    speakers: List[SpeakerSegment]
    associations: List[Association]
    artifacts: Artifacts
    model_versions: ModelVersions

    model_config = {"protected_namespaces": ()}


class SSEEvent(BaseModel):
    event: str
    data: Dict[str, Any]
