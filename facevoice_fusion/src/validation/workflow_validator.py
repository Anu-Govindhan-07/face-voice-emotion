from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import FACE_MATCH_THRESHOLD, IDENTITY_STORE, JOBS_DIR, MODEL_VERSIONS


@dataclass
class ValidationResult:
    stage: str
    passed: bool
    message: str


@dataclass
class WorkflowValidatorConfig:
    face_detector_name: str = MODEL_VERSIONS.get("face_detector", "facenet-pytorch-mtcnn")
    face_embedder_name: str = MODEL_VERSIONS.get("face_embedder", "facenet-pytorch-inceptionresnetv1")
    emotion_model_name: str = MODEL_VERSIONS.get("emotion_model", "trpakov/vit-face-expression")
    match_threshold: float = FACE_MATCH_THRESHOLD
    embeddings_rel: str = "embeddings/face"
    diarization_rel: str = "diarization.json"
    transcript_rel: str = "transcript.json"
    emotions_rel: str = "emotions.json"
    associations_rel: str = "associations.json"
    ui_rel: str = "ui.json"
    audio_rel: str = "audio.wav"
    faces_rel: str = "faces_tracks.json"
    identity_store_path: Path = IDENTITY_STORE


@dataclass
class ValidationReport:
    job_id: str
    results: List[ValidationResult]

    @property
    def has_failures(self) -> bool:
        return any(not row.passed for row in self.results)

    def to_text(self) -> str:
        lines = [f"Workflow Validation Report for job_id: {self.job_id}"]
        for row in self.results:
            status = "PASS" if row.passed else "FAIL"
            lines.append(f"[{status}] {row.stage} -> {row.message}")
        return "\n".join(lines)


class WorkflowValidator:
    def __init__(self, config: WorkflowValidatorConfig) -> None:
        self.config = config

    def validate(self, job_id: str, video_path: Path, strict: bool = False) -> ValidationReport:
        job_dir = JOBS_DIR / job_id
        results: List[ValidationResult] = []

        results.append(self._validate_entrypoint())
        results.append(self._validate_face_detection_and_tracks(job_dir))
        results.append(self._validate_face_embeddings(job_dir))
        results.append(self._validate_emotion(job_dir))
        results.append(self._validate_audio_diarization_transcription(job_dir))
        results.append(self._validate_name_detection(job_dir))
        results.append(self._validate_identity_store(job_dir))
        results.append(self._validate_cross_video_matching(job_dir))
        results.append(self._validate_artifacts(job_dir))
        results.append(self._validate_step_order(job_dir))

        if strict:
            for row in results:
                if "not found" in row.message.lower() or "missing" in row.message.lower():
                    row.passed = False

        return ValidationReport(job_id=job_id, results=results)

    def _validate_entrypoint(self) -> ValidationResult:
        app_main = Path(__file__).resolve().parents[2] / "app" / "main.py"
        pipeline_file = Path(__file__).resolve().parents[2] / "pipeline" / "run_pipeline.py"
        if not app_main.exists() or not pipeline_file.exists():
            return ValidationResult("Video upload entry point", False, "main.py or run_pipeline.py not found")
        upload_src = app_main.read_text()
        pipe_src = pipeline_file.read_text()
        links_to_pipeline = "background_tasks.add_task(run_pipeline" in upload_src
        audio_branch = "extract_audio(" in pipe_src
        video_branch = "detect_and_track(" in pipe_src
        if links_to_pipeline and audio_branch and video_branch:
            return ValidationResult(
                "Video upload entry point",
                True,
                "UI upload handler starts run_pipeline and triggers audio + video branches",
            )
        return ValidationResult(
            "Video upload entry point",
            False,
            "upload handler/pipeline link missing or one of extract_audio/detect_and_track is absent",
        )

    def _validate_face_detection_and_tracks(self, job_dir: Path) -> ValidationResult:
        detector_src = (Path(__file__).resolve().parents[2] / "pipeline" / "face_detect_track.py").read_text()
        uses_mtcnn = "from facenet_pytorch import MTCNN" in detector_src and "MTCNN(" in detector_src
        tracks_path = job_dir / self.config.faces_rel
        if not tracks_path.exists():
            return ValidationResult("Face detection + tracking (MTCNN)", False, f"Missing {tracks_path}")
        tracks = json.loads(tracks_path.read_text()).get("tracks", [])
        valid_tracks = [
            t
            for t in tracks
            if t.get("track_id") and isinstance(t.get("bboxes"), list) and all("t" in b for b in t.get("bboxes", []))
        ]
        if uses_mtcnn and valid_tracks:
            return ValidationResult("Face detection + tracking (MTCNN)", True, f"tracks={len(valid_tracks)}")
        return ValidationResult(
            "Face detection + tracking (MTCNN)",
            False,
            "MTCNN usage not found in code or track_id/timestamp fields missing in face tracks",
        )

    def _validate_face_embeddings(self, job_dir: Path) -> ValidationResult:
        embed_file = Path(__file__).resolve().parents[2] / "pipeline" / "face_embed.py"
        src = embed_file.read_text()
        uses_model = "InceptionResnetV1" in src and 'pretrained="vggface2"' in src
        emb_dir = job_dir / self.config.embeddings_rel
        emb_files = sorted(emb_dir.glob("*.npy")) if emb_dir.exists() else []
        if not emb_files:
            return ValidationResult("Face embeddings (InceptionResnetV1 vggface2)", False, f"No .npy files in {emb_dir}")
        shapes = []
        for emb_file in emb_files:
            try:
                arr = np.load(emb_file)
                shapes.append(tuple(arr.shape))
            except Exception as exc:
                return ValidationResult("Face embeddings (InceptionResnetV1 vggface2)", False, f"Failed loading {emb_file}: {exc}")
        one_dim = all(len(s) == 1 for s in shapes)
        same_dim = len({s[0] for s in shapes if s}) == 1
        if uses_model and one_dim and same_dim:
            return ValidationResult(
                "Face embeddings (InceptionResnetV1 vggface2)",
                True,
                f"N embeddings={len(emb_files)}, dim={shapes[0][0]}",
            )
        return ValidationResult("Face embeddings (InceptionResnetV1 vggface2)", False, "Model signature or embedding shape consistency failed")

    def _validate_emotion(self, job_dir: Path) -> ValidationResult:
        emotions_path = job_dir / self.config.emotions_rel
        if not emotions_path.exists():
            return ValidationResult("Emotion detection (vit-face-expression)", False, f"Missing {emotions_path}")
        payload = json.loads(emotions_path.read_text())
        tracks = payload.get("tracks", {})
        has_conf = False
        intervals: List[float] = []
        for row in tracks.values():
            timeline = row.get("timeline", [])
            for item in timeline:
                if "conf" in item:
                    has_conf = True
                if "start" in item and "end" in item:
                    delta = float(item["end"]) - float(item["start"])
                    if delta > 0:
                        intervals.append(delta)
        approx_interval = median(intervals) if intervals else 0.0
        model_ok = "vit-face-expression" in self.config.emotion_model_name
        if model_ok and has_conf:
            return ValidationResult(
                "Emotion detection (vit-face-expression)",
                True,
                f"tracks={len(tracks)}, median_interval={approx_interval:.2f}s",
            )
        return ValidationResult("Emotion detection (vit-face-expression)", False, "Model name mismatch or emotion confidence missing in timeline")

    def _validate_audio_diarization_transcription(self, job_dir: Path) -> ValidationResult:
        audio_path = job_dir / self.config.audio_rel
        diar_path = job_dir / self.config.diarization_rel
        transcript_path = job_dir / self.config.transcript_rel
        missing = [str(p) for p in [audio_path, diar_path, transcript_path] if not p.exists()]
        if missing:
            return ValidationResult("Audio → Diarization → Transcription", False, f"Missing artifacts: {', '.join(missing)}")

        diar_segments = json.loads(diar_path.read_text()).get("segments", [])
        tr_segments = json.loads(transcript_path.read_text()).get("segments", [])
        diar_ok = all("speaker_id" in s and "start" in s and "end" in s for s in diar_segments)
        aligned = bool(tr_segments) and all("speaker_id" in s for s in tr_segments if s.get("text"))
        if diar_ok and aligned:
            speakers = sorted({s.get("speaker_id") for s in diar_segments})
            return ValidationResult("Audio → Diarization → Transcription", True, f"speakers={speakers}, transcript_segments={len(tr_segments)}")
        return ValidationResult("Audio → Diarization → Transcription", False, "Diarization schema invalid or transcript not aligned with speaker IDs")

    def _validate_name_detection(self, job_dir: Path) -> ValidationResult:
        assoc_path = job_dir / self.config.associations_rel
        if not assoc_path.exists():
            return ValidationResult("Name detection", False, f"Missing {assoc_path}")
        payload = json.loads(assoc_path.read_text())
        events = payload.get("event_log", [])
        name_events = [
            evt
            for evt in events
            if evt.get("type") in {"self_intro", "mention"} and evt.get("name") and "speaker_id" in evt and "ts" in evt
        ]
        has_types = {evt.get("type") for evt in name_events}
        if name_events and has_types.intersection({"self_intro", "mention"}):
            return ValidationResult("Name detection", True, f"events={len(name_events)}, types={sorted(has_types)}")
        return ValidationResult("Name detection", False, "No diarization/transcription-linked self/mention name events found")

    def _validate_identity_store(self, job_dir: Path) -> ValidationResult:
        store_path = self.config.identity_store_path
        if not store_path.exists():
            return ValidationResult("Identity store update", False, f"Missing {store_path}")
        payload = json.loads(store_path.read_text())
        identities = payload.get("identities", payload.get("persons", []))
        if isinstance(identities, dict):
            identities = list(identities.values())
        schema_ok = all(
            isinstance(row, dict) and row.get("id") and row.get("name") and isinstance(row.get("embeddings", []), list)
            for row in identities
        )
        has_embedding_file = any(Path(str(emb)).exists() for row in identities for emb in row.get("embeddings", []))

        assoc_path = job_dir / self.config.associations_rel
        labels = []
        if assoc_path.exists():
            labels = [t for t in json.loads(assoc_path.read_text()).get("tracks", []) if t.get("label") not in {None, "", "Unknown"}]
        if schema_ok and has_embedding_file:
            msg = f"identities={len(identities)}"
            if labels:
                msg += ", named tracks present"
            return ValidationResult("Identity store update", True, msg)
        return ValidationResult("Identity store update", False, "identities.json schema invalid or embedding files not persisted")

    def _validate_cross_video_matching(self, job_dir: Path) -> ValidationResult:
        identity_file = Path(__file__).resolve().parents[2] / "pipeline" / "identity.py"
        run_file = Path(__file__).resolve().parents[2] / "pipeline" / "run_pipeline.py"
        identity_src = identity_file.read_text()
        run_src = run_file.read_text()
        uses_cosine = "_cosine_similarity" in identity_src and "np.dot" in identity_src
        has_unknown_logic = '"Unknown"' in run_src and 'status"' in run_src
        assoc_path = job_dir / self.config.associations_rel
        has_ui_labels = False
        if assoc_path.exists():
            tracks = json.loads(assoc_path.read_text()).get("tracks", [])
            has_ui_labels = all("label" in t for t in tracks) if tracks else False
        if uses_cosine and has_unknown_logic and has_ui_labels:
            return ValidationResult("Cross-video matching", True, "embedding similarity + Unknown fallback + UI labels present")
        return ValidationResult("Cross-video matching", False, "matching logic or UI label output missing")

    def _validate_artifacts(self, job_dir: Path) -> ValidationResult:
        required = [
            job_dir / self.config.embeddings_rel,
            job_dir / self.config.diarization_rel,
            job_dir / self.config.transcript_rel,
            job_dir / self.config.emotions_rel,
            job_dir / self.config.associations_rel,
            self.config.identity_store_path,
        ]
        missing: List[str] = []
        for artifact in required:
            if artifact.name == Path(self.config.embeddings_rel).name:
                if not artifact.exists() or not list(artifact.glob("*.npy")):
                    missing.append(f"{artifact}/*.npy")
            elif not artifact.exists():
                missing.append(str(artifact))
        if missing:
            return ValidationResult("Artifact assertions", False, f"Missing required outputs: {', '.join(missing)}")
        return ValidationResult("Artifact assertions", True, "All required output artifacts exist")

    def _validate_step_order(self, job_dir: Path) -> ValidationResult:
        ordered_files = {
            "diarization": job_dir / self.config.diarization_rel,
            "transcript": job_dir / self.config.transcript_rel,
            "associations": job_dir / self.config.associations_rel,
            "ui": job_dir / self.config.ui_rel,
        }
        embeddings_dir = job_dir / self.config.embeddings_rel
        emb_files = sorted(embeddings_dir.glob("*.npy")) if embeddings_dir.exists() else []
        if not emb_files:
            return ValidationResult("Order and dependencies", False, "No embeddings available for dependency checks")
        missing = [name for name, path in ordered_files.items() if not path.exists()]
        if missing:
            return ValidationResult("Order and dependencies", False, f"Missing files for order check: {missing}")

        emb_time = max(p.stat().st_mtime for p in emb_files)
        diar_t = ordered_files["diarization"].stat().st_mtime
        tr_t = ordered_files["transcript"].stat().st_mtime
        assoc_t = ordered_files["associations"].stat().st_mtime
        ui_t = ordered_files["ui"].stat().st_mtime

        violations: List[str] = []
        if diar_t > assoc_t:
            violations.append("diarization after name detection")
        if tr_t > assoc_t:
            violations.append("transcription after name detection")
        if emb_time > assoc_t:
            violations.append("embeddings after identity matching")
        if assoc_t > ui_t:
            violations.append("identity matching after final UI tag assignment")

        if violations:
            return ValidationResult("Order and dependencies", False, "; ".join(violations))
        return ValidationResult("Order and dependencies", True, "Artifact creation times satisfy dependency ordering")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate end-to-end workflow implementation and artifacts")
    parser.add_argument("--job_id", required=True, help="Job ID to validate")
    parser.add_argument("--video", required=True, help="Path to input video used for the job")
    parser.add_argument("--strict", action="store_true", help="Treat any weak signal as failure")
    parser.add_argument("--emotion-model-name", default=None)
    parser.add_argument("--face-detector-name", default=None)
    parser.add_argument("--face-embedder-name", default=None)
    parser.add_argument("--identity-store", default=None)
    parser.add_argument("--match-threshold", type=float, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = WorkflowValidatorConfig()
    if args.emotion_model_name:
        cfg.emotion_model_name = args.emotion_model_name
    if args.face_detector_name:
        cfg.face_detector_name = args.face_detector_name
    if args.face_embedder_name:
        cfg.face_embedder_name = args.face_embedder_name
    if args.identity_store:
        cfg.identity_store_path = Path(args.identity_store)
    if args.match_threshold is not None:
        cfg.match_threshold = args.match_threshold

    validator = WorkflowValidator(cfg)
    report = validator.validate(job_id=args.job_id, video_path=Path(args.video), strict=args.strict)
    print(report.to_text())

    return 1 if report.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
