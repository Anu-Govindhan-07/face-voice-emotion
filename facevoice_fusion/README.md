# Face Voice Emotion Fusion with Re-Identification

A multimodal AI pipeline for post-production workflows that detects **who** appears or speaks in a video and estimates **how they feel** by combining **facial emotion**, **vocal emotion**, and **identity persistence across scenes**.

---

## 1. Project Overview

### 1.1 What this project does
This project processes video and audio together to answer two production-critical questions:

1. **Who is on screen / who is speaking?**
2. **What emotion is that person expressing over time?**

The system combines:
- **Face analysis** for visual emotion signals
- **Speech analysis** for vocal emotion signals
- **Re-identification (ReID)** to keep the same identity across shots, cuts, and re-entries
- **Fusion logic** to combine face and voice predictions into one final emotion timeline

### 1.2 Why this project matters
In real production footage, emotion is not always reliably visible from a single modality:
- A face may be partially hidden, side-facing, blurred, or under difficult lighting
- Audio may contain noise, music, overlapping dialogue, or off-screen speakers
- Traditional tracking often breaks across cuts or scene changes

By combining multiple signals, this project creates a more reliable and editor-friendly emotional analysis pipeline.

---

## 2. Business Value for a Post-Production Company

This system can support post-production teams in several ways:

### 2.1 Faster content search
Editors can search footage using metadata such as:
- "show angry reaction shots"
- "find sad dialogue from Character A"
- "find emotional peaks in this scene"

### 2.2 Better take selection
The system can help compare takes based on:
- emotion intensity
- emotion consistency
- alignment between dialogue tone and facial expression

### 2.3 Character emotion timeline
It can generate a timeline of emotional states per character across scenes or episodes, which helps:
- story continuity
- trailer editing
- recap generation
- scene analysis

### 2.4 Metadata generation
Outputs can be exported to:
- JSON
- CSV
- timeline markers
- media asset management systems (MAM)

### 2.5 Quality control and continuity
The system can flag:
- emotion mismatch between face and voice
- sudden identity switches
- unstable tracking across cuts

---

## 3. Core Features

- Face detection in video frames
- Face tracking inside shots
- Face re-identification across scenes
- Speaker diarization from audio
- Voice emotion recognition
- Face emotion recognition
- Face-voice alignment
- Emotion fusion with confidence weighting
- Temporal smoothing for stable predictions
- Structured export for downstream editing tools

---

## 4. High-Level Workflow

1. Load input video
2. Extract frames and audio
3. Detect faces in frames
4. Track faces within local shot windows
5. Generate face embeddings for identity matching
6. Re-identify the same person across cuts
7. Segment audio by speaker turns
8. Predict emotion from face tracks
9. Predict emotion from speech segments
10. Align face tracks with speaker turns
11. Fuse predictions from both modalities
12. Smooth predictions over time
13. Export results and summaries

---

## 5. System Architecture

```text
Input Video
   |
   +--> Frame Extraction
   |       |
   |       +--> Face Detection
   |       +--> Face Tracking
   |       +--> Face Embeddings
   |       +--> Face Emotion
   |       +--> Re-Identification
   |
   +--> Audio Extraction
           |
           +--> Speaker Diarization
           +--> Speech Segmentation
           +--> Voice Emotion

Face + Voice Alignment
           |
           +--> Fusion Layer
           +--> Temporal Smoothing
           +--> Final Per-Person Emotion Timeline
           +--> JSON / CSV / Visualization Output
```

---

## 6. Models Used

> Note: The exact model names in your implementation may differ. The list below reflects a production-ready reference stack for this project.

### 6.1 Face Detection
Recommended models:
- **RetinaFace**
- **SCRFD**
- **YOLO-based face detector**

**Purpose:** Detect face bounding boxes and facial landmarks in each frame.

**Why used:**
- Strong performance under pose variation
- Good robustness on real-world footage
- Fast enough for batch inference

### 6.2 Face Tracking
Common approaches:
- SORT / DeepSORT
- IoU-based temporal association
- Tracklet generation per shot

**Purpose:** Maintain short-term continuity of the same face within nearby frames.

### 6.3 Face Re-Identification
Recommended models:
- **ArcFace**
- **CurricularFace**
- **MagFace**

**Primary choice:** **ArcFace**

**Purpose:** Generate identity embeddings so the same person can be matched across shots and scene cuts.

**Why ArcFace:**
- Very strong identity discrimination
- Efficient inference
- Well-supported in face recognition pipelines
- Performs reliably for clustering and cross-shot matching

### 6.4 Face Emotion Recognition
Recommended model families:
- **ResNet-based classifiers**
- **EfficientNet-based classifiers**
- Models fine-tuned on **AffectNet**, **RAF-DB**, or **FER2013**

**Purpose:** Predict facial emotion from cropped face frames.

Common emotion classes:
- neutral
- happy
- sad
- angry
- fear
- disgust
- surprise
- contempt (optional)
- calm (optional, custom label)

### 6.5 Speaker Diarization
Recommended tool:
- **pyannote.audio**

**Purpose:** Detect speaker turns and segment audio into who-spoke-when regions.

**Why used:**
- Strong out-of-the-box diarization quality
- Helpful for matching emotion to the active speaker
- Good community support

### 6.6 Voice Emotion Recognition
Recommended model families:
- **Wav2Vec2**
- **HuBERT**
- wav2vec-style SSL speech encoders fine-tuned for SER (speech emotion recognition)

**Primary choice:** **Wav2Vec2 / HuBERT fine-tuned for SER**

**Purpose:** Predict emotion from speech segments using prosody, tone, and acoustic patterns.

**Why used:**
- Better representation learning than handcrafted audio features
- Strong performance on downstream speech tasks
- More robust than MFCC + classical ML for complex emotional patterns

### 6.7 Fusion Model
Options:
- Rule-based late fusion
- Weighted score fusion
- Confidence-gated fusion
- Small MLP over modality scores
- Attention-based multimodal fusion

**Primary choice:** **Confidence-weighted late fusion**

**Purpose:** Combine face emotion and voice emotion into a single final decision.

**Why used:**
- Easy to debug
- Easier to deploy than full multimodal transformers
- More practical when labeled multimodal training data is limited
- Works well for production pipelines with gradual improvements

---

## 7. Why These Models Were Chosen

### 7.1 Selection criteria
Models were chosen based on:
- accuracy
- robustness on real footage
- inference speed
- ease of integration
- pretrained availability
- deployment simplicity
- stability under noise, lighting changes, and scene cuts

### 7.2 Why not only face emotion?
Face-only systems fail when:
- the face is occluded
- the person is off-angle
- lighting is poor
- the shot is too short
- the emotion is expressed more strongly in voice than expression

### 7.3 Why not only voice emotion?
Voice-only systems fail when:
- background music dominates
- multiple speakers overlap
- the speaker is off-screen
- speech is too short or too quiet
- the emotion is visually present but not strongly audible

### 7.4 Why not use a large end-to-end multimodal transformer?
Although large multimodal transformers can be powerful, they were not selected as the default option because:
- they require more labeled multimodal data
- training and tuning are more expensive
- deployment is heavier
- interpretability is lower
- iteration speed is slower in production environments

---

## 8. Alternatives Considered and Why They Were Not Used

### 8.1 Face ReID alternatives
**FaceNet**
- Good baseline, but often outperformed by ArcFace in identity discrimination

**SphereFace**
- Strong academic model, but less common in current production stacks

**MagFace**
- Strong alternative, but ArcFace often has broader ecosystem support and easier integration

### 8.2 Voice emotion alternatives
**MFCC + SVM / RandomForest**
- Lightweight, but generally weaker than self-supervised speech embeddings
- Less robust to complex, noisy post-production audio

**OpenSMILE feature engineering**
- Useful for classical pipelines, but requires heavier manual feature tuning

**Whisper embeddings**
- Excellent for ASR, but not always the best default emotion representation without task-specific tuning

### 8.3 Fusion alternatives
**Early fusion**
- Combines raw or intermediate features before classification
- Not selected because it needs stronger synchronized multimodal training data

**Cross-attention multimodal transformer**
- Powerful but heavier and harder to train
- Higher engineering and infrastructure cost

---

## 9. Suggested Project Structure

```text
face-voice-emotion-fusion/
|
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- default.yaml
|   |-- models.yaml
|
|-- data/
|   |-- input/
|   |-- output/
|   |-- temp/
|
|-- models/
|   |-- face_detector/
|   |-- face_reid/
|   |-- face_emotion/
|   |-- diarization/
|   |-- voice_emotion/
|
|-- src/
|   |-- main.py
|   |-- pipeline.py
|   |-- config.py
|   |
|   |-- video/
|   |   |-- extract_frames.py
|   |   |-- detect_faces.py
|   |   |-- track_faces.py
|   |   |-- face_emotion.py
|   |   |-- reid.py
|   |
|   |-- audio/
|   |   |-- extract_audio.py
|   |   |-- diarize.py
|   |   |-- voice_emotion.py
|   |
|   |-- fusion/
|   |   |-- align_modalities.py
|   |   |-- fuse_predictions.py
|   |   |-- smooth.py
|   |
|   |-- utils/
|   |   |-- io.py
|   |   |-- logger.py
|   |   |-- visualization.py
|
|-- notebooks/
|-- scripts/
|   |-- run_demo.sh
|   |-- evaluate.sh
|
|-- outputs/
|   |-- json/
|   |-- csv/
|   |-- plots/
```

---

## 10. Environment Setup

### 10.1 Prerequisites
Install the following:
- Python 3.10 or 3.11
- pip
- ffmpeg
- git
- optional: CUDA-enabled GPU for faster inference

### 10.2 Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 10.3 Install dependencies
```bash
pip install -r requirements.txt
```

### 10.4 Example requirements.txt
```text
numpy
pandas
opencv-python
torch
torchvision
torchaudio
scikit-learn
scipy
matplotlib
tqdm
PyYAML
ffmpeg-python
librosa
soundfile
moviepy
facenet-pytorch
transformers
pyannote.audio
onnxruntime
```

> Add or remove packages based on your actual implementation.

---

## 11. Model Setup

Create a directory for model weights:

```bash
mkdir -p models
```

Example layout:
```text
models/
|-- face_detector/
|   |-- retinaface.onnx
|
|-- face_reid/
|   |-- arcface.onnx
|
|-- face_emotion/
|   |-- affectnet_resnet50.pt
|
|-- diarization/
|   |-- pyannote_config.yaml
|
|-- voice_emotion/
|   |-- wav2vec2_ser.pt
```

### 11.1 Hugging Face / pyannote note
Some diarization or speech models may require:
- a Hugging Face account
- authentication token
- license acceptance for specific checkpoints

Example:
```bash
huggingface-cli login
```

---

## 12. Configuration

Example `configs/default.yaml`:

```yaml
input_video: data/input/sample.mp4
output_dir: data/output/run_01

video:
  fps_sample: 5
  frame_size: 640

face_detection:
  model: retinaface
  conf_threshold: 0.6

tracking:
  max_age: 30
  min_hits: 3

reid:
  model: arcface
  similarity_threshold: 0.45
  clustering: agglomerative

audio:
  sample_rate: 16000

diarization:
  enabled: true

emotion:
  face_model: affectnet_resnet50
  voice_model: wav2vec2_ser
  labels: [neutral, happy, sad, angry, fear, disgust, surprise]

fusion:
  method: confidence_weighted
  face_weight: 0.55
  voice_weight: 0.45
  temporal_smoothing: true
```

---

## 13. How to Run the Project

> Update the command names if your repository uses different entry points.

### 13.1 Basic run
```bash
python src/main.py --config configs/default.yaml
```

### 13.2 Run on a specific video
```bash
python src/main.py   --input data/input/scene_01.mp4   --output data/output/scene_01   --face-detector retinaface   --face-reid arcface   --face-emotion affectnet_resnet50   --voice-emotion wav2vec2_ser
```

### 13.3 Run with GPU
```bash
python src/main.py --config configs/default.yaml --device cuda
```

### 13.4 Run only visual pipeline
```bash
python src/main.py --config configs/default.yaml --disable-audio
```

### 13.5 Run only audio pipeline
```bash
python src/main.py --config configs/default.yaml --disable-video
```

### 13.6 Run evaluation
```bash
python src/evaluate.py --pred outputs/json/predictions.json --gt data/labels/ground_truth.json
```

---

## 14. Example End-to-End Execution Flow

When you run the pipeline, the following happens:

1. The input video is loaded
2. Frames are extracted at the configured sampling rate
3. Audio is extracted and resampled
4. Faces are detected frame by frame
5. Face tracklets are formed over time
6. Identity embeddings are generated for each tracklet
7. ReID clusters link the same person across cuts
8. Speaker diarization splits audio by speaker turns
9. Face emotion predictions are computed per face track
10. Voice emotion predictions are computed per speaker segment
11. Tracks and speaker turns are aligned by time
12. Fusion logic combines both modalities
13. Temporal smoothing reduces jitter
14. Final outputs are saved to disk

---

## 15. Input and Output Format

### 15.1 Input
Supported inputs:
- `.mp4`
- `.mov`
- `.mkv`
- `.wav` (audio-only mode, if implemented)

### 15.2 Output files
Typical outputs:
- `predictions.json`
- `predictions.csv`
- `track_visualization.mp4`
- `emotion_timeline.png`
- `identity_summary.json`

### 15.3 Example JSON output
```json
{
  "video_id": "scene_01",
  "persons": [
    {
      "person_id": "P001",
      "segments": [
        {
          "start": 3.20,
          "end": 6.85,
          "face_emotion": "sad",
          "voice_emotion": "sad",
          "final_emotion": "sad",
          "confidence": 0.91
        },
        {
          "start": 8.10,
          "end": 10.30,
          "face_emotion": "angry",
          "voice_emotion": "neutral",
          "final_emotion": "angry",
          "confidence": 0.72
        }
      ]
    }
  ]
}
```

---

## 16. How Fusion Works

### 16.1 Late fusion
The default implementation uses **late fusion**:
- predict face emotion independently
- predict voice emotion independently
- combine scores at the decision level

### 16.2 Confidence-weighted decision
Example:
```text
final_score = (face_weight * face_score * face_confidence) +
              (voice_weight * voice_score * voice_confidence)
```

### 16.3 Dynamic weighting logic
Possible weighting strategy:
- if face visibility is low -> increase voice weight
- if audio quality is low -> increase face weight
- if speaker is off-screen -> voice only
- if no speech exists -> face only

### 16.4 Temporal smoothing
A smoothing layer helps reduce unstable frame-by-frame jumps:
- moving average
- majority vote in a window
- HMM-based smoothing
- confidence thresholding

---

## 17. Re-Identification Logic

The ReID step ensures identity consistency across scene cuts.

### 17.1 Process
- crop faces from tracklets
- compute embeddings using ArcFace
- compare embeddings via cosine similarity
- cluster similar identities
- assign global IDs such as `P001`, `P002`, etc.

### 17.2 Why ReID is necessary
Simple trackers are local in time and often fail when:
- cuts happen
- a person leaves and re-enters
- camera angle changes drastically
- lighting changes between shots

ReID solves this by focusing on identity embedding similarity rather than only short-term motion tracking.

---

## 18. Evaluation

### 18.1 Emotion metrics
- accuracy
- macro F1
- weighted F1
- confusion matrix
- segment-level consistency

### 18.2 ReID metrics
- precision / recall
- cluster purity
- ID switch count
- false merge / false split rates

### 18.3 Runtime metrics
- processing time per minute of footage
- GPU memory usage
- average time per module

---

## 19. Limitations

- overlapping speakers reduce diarization quality
- heavy background music affects speech emotion
- occluded or profile faces reduce visual reliability
- emotion labels may be subjective
- datasets may not perfectly match cinematic footage
- sarcasm and context-driven emotion remain difficult
- subtle emotions are harder than strong emotions

---

## 20. Recommendations for Production Use

- Use GPU inference for long-form footage
- Calibrate fusion weights on internal studio data
- Keep a manual review workflow for critical editorial decisions
- Add shot boundary detection for cleaner track segmentation
- Fine-tune emotion models on domain-specific footage if possible
- Store outputs in a searchable metadata database

---

## 21. Troubleshooting

### Problem: No faces detected
Possible causes:
- low light
- extreme pose
- detector threshold too high

Try:
- lowering face detection threshold
- increasing frame resolution
- switching to a stronger face detector

### Problem: Identity switches across cuts
Possible causes:
- weak ReID threshold
- poor face crops
- motion blur

Try:
- using better crops
- tuning similarity threshold
- aggregating embeddings over multiple frames

### Problem: Voice emotion is unstable
Possible causes:
- overlapping speakers
- short speech segments
- strong background music

Try:
- minimum speech duration filtering
- denoising
- stronger diarization configuration

### Problem: Fused output is jittery
Try:
- enabling temporal smoothing
- increasing segment window size
- using confidence thresholds before label switching

---

## 22. Future Improvements

- subtitle sentiment integration
- body pose / gesture emotion cues
- emotion intensity regression
- scene-level group emotion estimation
- multimodal transformer fusion
- active learning with human feedback
- direct plugin export to editing systems

---

## 23. Security, Privacy, and Ethical Notes

Because this system analyzes identity and emotion, it should be used carefully.

Recommendations:
- process only authorized footage
- protect exported metadata
- avoid using emotion predictions as absolute truth
- keep human review in the loop
- document model limitations and failure modes
- follow internal legal and privacy requirements

---

## 24. Demo Script for Team Presentation

You can present the system using the following flow:

1. Show an input clip
2. Show face boxes and person IDs
3. Show diarization timeline
4. Show face and voice emotion predictions
5. Show the fused final emotion result
6. Show exported timeline/JSON
7. Explain where the system helps editors

---

## 25. Sample Commands

### Extract audio manually
```bash
ffmpeg -i data/input/sample.mp4 -ar 16000 -ac 1 data/temp/audio.wav
```

### Save frames manually
```bash
ffmpeg -i data/input/sample.mp4 data/temp/frames/frame_%05d.jpg
```

### Run a demo shell script
```bash
bash scripts/run_demo.sh
```

Example `run_demo.sh`:
```bash
python src/main.py   --input data/input/sample.mp4   --output data/output/sample_run   --device cuda
```

---

## 26. Conclusion

Face Voice Emotion Fusion with Re-Identification is a practical multimodal system for post-production intelligence.

It combines:
- identity continuity
- facial emotion
- vocal emotion
- fusion-based robustness

This makes it useful for:
- emotion-aware search
- editorial assistance
- take selection
- character analysis
- metadata generation
- continuity support

---

## 27. Acknowledgements

Typical external building blocks may include:
- PyTorch
- pyannote.audio
- Hugging Face Transformers
- OpenCV
- ffmpeg
- face recognition / FER model libraries

---

