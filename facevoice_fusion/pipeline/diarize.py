from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.config import HUGGINGFACE_TOKEN
from .utils import console, save_json


# ----------------------------
# Math helpers
# ----------------------------
def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    sim = float(np.dot(a, b) / denom)
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def _kmeans(features: np.ndarray, k: int, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple k-means for small datasets. Features are expected to be normalized.
    """
    if len(features) < k:
        raise ValueError("not enough samples for kmeans")
    rng = np.random.default_rng(42)
    centroids = features[rng.choice(len(features), size=k, replace=False)].copy()
    labels = np.zeros(len(features), dtype=np.int64)

    for _ in range(max_iter):
        dists = np.stack([np.linalg.norm(features - c, axis=1) for c in centroids], axis=1)
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
                centroids[idx] = centroids[idx] / (np.linalg.norm(centroids[idx]) + 1e-8)

    return labels, centroids


def _cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return np.clip(x @ x.T, -1.0, 1.0)


def _silhouette_score_cosine(x: np.ndarray, labels: np.ndarray) -> float:
    """
    Lightweight silhouette score approximation using cosine distance.
    Returns [-1..1] (higher is better).
    """
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0

    sim = _cosine_sim_matrix(x)
    dist = 1.0 - sim

    s_vals: List[float] = []
    for i in range(len(x)):
        same = labels == labels[i]
        if same.sum() <= 1:
            continue
        a = float(dist[i, same].sum() / (same.sum() - 1))  # intra-cluster avg dist
        b_candidates = []
        for c in uniq:
            if c == labels[i]:
                continue
            mask = labels == c
            if mask.sum() == 0:
                continue
            b_candidates.append(float(dist[i, mask].mean()))
        if not b_candidates:
            continue
        b = min(b_candidates)
        s = (b - a) / max(a, b, 1e-8)
        s_vals.append(float(s))

    if not s_vals:
        return -1.0
    return float(np.mean(s_vals))


# ----------------------------
# Improved fallback diarization
# ----------------------------
def _fallback_multi_speaker_diarization(audio_path: Path) -> List[dict]:
    """
    Improved local diarization fallback:
    - Resample to 16k mono
    - Window audio into overlapping chunks
    - Simple VAD gating by energy (keeps speech-like windows)
    - Prefer speaker embeddings (SpeechBrain ECAPA) for clustering
    - Fallback to MFCC + deltas if SpeechBrain isn't available
    - Select k by cosine silhouette score (reduces collapse into single speaker)
    - Merge labeled windows into continuous speaker turns
    """

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        return []

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16k
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    total = waveform.shape[1]
    duration = total / sample_rate
    if duration < 1.0:
        return [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.25}]

    # Windowing (better for multi-speaker than your previous settings)
    window_s = 2.0
    hop_s = 0.5
    win = int(window_s * sample_rate)
    hop = int(hop_s * sample_rate)

    windows: List[Tuple[float, float, torch.Tensor]] = []
    energies: List[float] = []

    for start in range(0, max(1, total - win + 1), hop):
        end = min(total, start + win)
        seg = waveform[:, start:end]
        if seg.shape[1] < int(0.7 * win):
            continue

        # Energy proxy for VAD
        e = float(torch.mean(seg.abs()).item())
        windows.append((start / sample_rate, end / sample_rate, seg))
        energies.append(e)

    if not windows:
        return [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.25}]

    # VAD-like gating: keep top ~70% energy windows (less aggressive)
    e_arr = np.asarray(energies, dtype=np.float32)
    thr = float(np.percentile(e_arr, 30))
    thr = max(thr, 0.0025)

    active_idx = [i for i, e in enumerate(energies) if e >= thr]
    if len(active_idx) < 3:
        return [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.30}]

    active_windows = [windows[i] for i in active_idx]

    # ----------------------------
    # Feature extraction
    # ----------------------------
    feats: Optional[np.ndarray] = None

    # Try SpeechBrain ECAPA embeddings (best fallback quality)
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )

        emb_list: List[np.ndarray] = []
        for _, _, seg in active_windows:
            wav = seg.squeeze(0).float()  # [time]
            with torch.no_grad():
                emb = classifier.encode_batch(wav.unsqueeze(0))  # [1, 1, D] or [1, D]
            emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            emb_list.append(emb)

        feats = np.stack(emb_list, axis=0)
        console.log(f"Fallback diarization: using ECAPA embeddings ({feats.shape[0]} windows).")

    except Exception as exc:
        console.log(f"ECAPA embeddings not available, using MFCC fallback. Reason: {exc}")

    # MFCC + deltas fallback (improved from previous)
    if feats is None:
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=30,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
        )

        feat_list: List[np.ndarray] = []
        for _, _, seg in active_windows:
            m = mfcc(seg).squeeze(0)  # [n_mfcc, frames]
            d1 = torchaudio.functional.compute_deltas(m)
            d2 = torchaudio.functional.compute_deltas(d1)

            vec = torch.cat(
                [
                    m.mean(dim=1), m.std(dim=1),
                    d1.mean(dim=1), d1.std(dim=1),
                    d2.mean(dim=1), d2.std(dim=1),
                ],
                dim=0,
            ).cpu().numpy().astype(np.float32)

            vec = vec / (np.linalg.norm(vec) + 1e-8)
            feat_list.append(vec)

        feats = np.stack(feat_list, axis=0)
        console.log(f"Fallback diarization: using MFCC+deltas ({feats.shape[0]} windows).")

    # ----------------------------
    # Choose k (#speakers) by silhouette score
    # ----------------------------
    max_k = min(6, len(feats))  # allow up to 6 speakers
    if max_k < 2:
        return [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.30}]

    best_k = 1
    best_labels = np.zeros(len(feats), dtype=np.int64)
    best_score = -1e9

    # Evaluate k from 2..max_k (k=1 is trivial)
    for k in range(2, max_k + 1):
        labels, _ = _kmeans(feats, k)
        score = _silhouette_score_cosine(feats, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # If separation is weak, keep single speaker BUT less conservative than before
    # Old code used 0.02 threshold; this was too strict and caused collapse to S1.
    if best_k == 1 or best_score < 0.10:
        console.log(f"Fallback diarization: weak separation (score={best_score:.3f}); using 1 speaker.")
        return [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.35}]

    console.log(f"Fallback diarization: detected ~{best_k} speakers (score={best_score:.3f}).")

    # ----------------------------
    # Convert window labels to segments
    # ----------------------------
    diarization: List[dict] = []
    cur_label = int(best_labels[0])
    cur_start, cur_end, _ = active_windows[0]

    for idx in range(1, len(active_windows)):
        ws, we, _ = active_windows[idx]
        lab = int(best_labels[idx])

        # merge if same label and close in time
        if lab == cur_label and ws <= cur_end + 0.25:
            cur_end = max(cur_end, we)
            continue

        diarization.append(
            {"speaker_id": f"S{cur_label + 1}", "start": float(cur_start), "end": float(cur_end), "conf": 0.60}
        )
        cur_label = lab
        cur_start, cur_end = ws, we

    diarization.append({"speaker_id": f"S{cur_label + 1}", "start": float(cur_start), "end": float(cur_end), "conf": 0.60})
    diarization.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))

    # Merge tiny gaps for same speaker
    merged: List[dict] = []
    for seg in diarization:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg["speaker_id"] == prev["speaker_id"] and seg["start"] <= prev["end"] + 0.30:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg)

    return merged


# ----------------------------
# Public API
# ----------------------------
def diarize_audio(audio_path: Path, output_path: Path) -> List[dict]:
    """
    Uses pyannote/speaker-diarization when HUGGINGFACE_TOKEN is available.
    Otherwise uses improved local fallback diarization.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diarization: List[dict] = []

    # Primary: pyannote (best quality if installed + token exists)
    if HUGGINGFACE_TOKEN:
        try:
            from pyannote.audio import Pipeline  # type: ignore

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=HUGGINGFACE_TOKEN,
            )
            diar = pipeline(str(audio_path))

            speaker_map: Dict[str, str] = {}
            next_idx = 1

            for turn, _, label in diar.itertracks(yield_label=True):
                if label not in speaker_map:
                    speaker_map[label] = f"S{next_idx}"
                    next_idx += 1
                diarization.append(
                    {
                        "speaker_id": speaker_map[label],
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "conf": 0.90,
                    }
                )

            diarization.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))

        except ImportError:
            console.log("Pyannote is not installed; falling back to local diarization.")
        except Exception as exc:
            console.log(f"Pyannote diarization failed, fallback mode: {exc}")

    # Fallback: improved local diarization
    if not diarization:
        console.log("Fallback diarization mode (improved local clustering)")
        diarization = _fallback_multi_speaker_diarization(audio_path)

        # absolute final fallback
        if not diarization:
            duration = sf.info(str(audio_path)).duration
            diarization = [{"speaker_id": "S1", "start": 0.0, "end": float(duration), "conf": 0.20}]

    save_json(output_path, {"segments": diarization})
    return diarization
