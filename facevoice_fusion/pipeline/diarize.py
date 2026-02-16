from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.config import HUGGINGFACE_TOKEN
from .utils import console, save_json


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    sim = float(np.dot(a, b) / denom)
    return 1.0 - max(-1.0, min(1.0, sim))


def _kmeans(features: np.ndarray, k: int, max_iter: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    if len(features) < k:
        raise ValueError("not enough samples for kmeans")
    rng = np.random.default_rng(42)
    centroids = features[rng.choice(len(features), size=k, replace=False)]
    labels = np.zeros(len(features), dtype=np.int64)

    for _ in range(max_iter):
        dists = np.stack([
            np.linalg.norm(features - c, axis=1) for c in centroids
        ], axis=1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = features[labels == idx]
            if len(members) == 0:
                centroids[idx] = features[rng.integers(0, len(features))]
            else:
                centroids[idx] = members.mean(axis=0)
    return labels, centroids


def _cluster_score(features: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    if len(np.unique(labels)) <= 1:
        return -1.0
    intra = float(np.mean([np.linalg.norm(features[i] - centroids[labels[i]]) for i in range(len(features))]))
    centroid_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            centroid_dists.append(_cosine_distance(centroids[i], centroids[j]))
    if not centroid_dists:
        return -1.0
    inter = float(np.mean(centroid_dists))
    return inter - intra


def _fallback_multi_speaker_diarization(audio_path: Path) -> List[dict]:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        return []
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    window_s = 1.5
    hop_s = 0.75
    win = int(window_s * sample_rate)
    hop = int(hop_s * sample_rate)
    total = waveform.shape[1]
    if total < win:
        duration = total / sample_rate
        return [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.25}]

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=20,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
    )

    windows: List[Tuple[float, float]] = []
    features: List[np.ndarray] = []
    energies: List[float] = []

    for start in range(0, max(1, total - win + 1), hop):
        end = min(total, start + win)
        seg = waveform[:, start:end]
        if seg.shape[1] < int(0.6 * win):
            continue
        energy = float(torch.mean(seg.abs()).item())
        energies.append(energy)
        windows.append((start / sample_rate, end / sample_rate))
        m = mfcc(seg).squeeze(0).numpy()
        feat = np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        features.append(feat)

    if not features:
        duration = sf.info(str(audio_path)).duration
        return [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.2}]

    feats = np.stack(features, axis=0)
    # keep only active speech-like windows
    energy_threshold = max(0.005, float(np.percentile(np.asarray(energies), 30)))
    active_idx = [i for i, e in enumerate(energies) if e >= energy_threshold]
    if len(active_idx) < 2:
        duration = sf.info(str(audio_path)).duration
        return [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.25}]

    active_feats = feats[active_idx]
    active_windows = [windows[i] for i in active_idx]

    best_k = 1
    best_labels = np.zeros(len(active_feats), dtype=np.int64)
    best_score = -1e9
    max_k = min(4, len(active_feats))
    for k in range(1, max_k + 1):
        labels, centroids = _kmeans(active_feats, k)
        score = _cluster_score(active_feats, labels, centroids) if k > 1 else -0.1
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # if separation is weak, keep single speaker
    if best_k == 1 or best_score < 0.02:
        duration = sf.info(str(audio_path)).duration
        return [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.3}]

    diarization: List[dict] = []
    start, end = active_windows[0]
    current = int(best_labels[0])
    for idx in range(1, len(active_windows)):
        ws, we = active_windows[idx]
        label = int(best_labels[idx])
        if label == current and ws <= end + 0.35:
            end = max(end, we)
            continue
        diarization.append({"speaker_id": f"S{current + 1}", "start": float(start), "end": float(end), "conf": 0.55})
        start, end, current = ws, we, label
    diarization.append({"speaker_id": f"S{current + 1}", "start": float(start), "end": float(end), "conf": 0.55})

    diarization.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))
    return diarization


def diarize_audio(audio_path: Path, output_path: Path) -> List[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diarization: List[dict] = []

    if HUGGINGFACE_TOKEN:
        try:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
            diar = pipeline(str(audio_path))
            speaker_map = {}
            next_idx = 1
            for turn, _, label in diar.itertracks(yield_label=True):
                if label not in speaker_map:
                    speaker_map[label] = f"S{next_idx}"
                    next_idx += 1
                diarization.append({
                    "speaker_id": speaker_map[label],
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "conf": 0.9,
                })
            diarization.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))
        except ImportError:
            console.log("Pyannote is not installed; falling back to local multi-speaker diarization.")
        except Exception as exc:
            console.log(f"Pyannote diarization failed, fallback mode: {exc}")

    if not diarization:
        console.log("Fallback diarization mode (local clustering)")
        diarization = _fallback_multi_speaker_diarization(audio_path)
        if not diarization:
            duration = sf.info(str(audio_path)).duration
            diarization = [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.2}]

    save_json(output_path, {"segments": diarization})
    return diarization
