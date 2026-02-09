# Speech → Name → Face Tagging Reliability Playbook

This playbook maps common production failure modes to practical fixes for the current FaceVoice Fusion pipeline.

## Guiding principles

1. **Keep confidence explicit**: output `matched` / `maybe` / `unknown` rather than forcing a hard label.
2. **Fuse multiple signals**: combine diarization, mouth motion/lip activity, time overlap, and identity priors.
3. **Prefer delayed commitment**: buffer a short temporal window before assigning names.
4. **Human-in-the-loop for enrollment**: never let low-confidence auto-enrollment poison the identity store.

## Issue-by-issue mitigations

### 1) Name mentioned but the person is not in the video

- Treat mention and self-identification as different event types.
- Keep `mentioned_name` on the association object but do not convert it to face identity unless visual confidence is high.
- If no plausible on-screen candidate exists, keep the name as **off-screen mention** metadata.

### 2) Multiple people visible when a single name is mentioned (ambiguous target)

- Score candidates with a weighted function:
  - temporal overlap with speaker segment,
  - speaking-face evidence (mouth motion),
  - face identity prior,
  - recent continuity with previous assignments.
- If the top two candidates are close, emit **ambiguous** and request review.

### 3) Off-screen speaker (audio speaker isn’t on camera)

- Add an `offscreen=true` state when no track reaches minimum association score.
- Preserve speaker diarization and transcript alignment, but skip face tag.

### 4) Overlapping speech / interruptions (diarization confusion)

- Use overlap-aware diarization output (multi-label speaker activity where supported).
- Segment associations at finer granularity (sub-second windows) instead of one label per long turn.

### 5) Diarization mistakes (speaker splits/merges, wrong speaker IDs)

- Add speaker embedding clustering post-pass to merge fragmented IDs.
- Add inconsistency checks (same speaker embedding appears under many IDs in short span) and auto-repair.

### 6) ASR errors for Swedish + names (mis-transcribed names, spelling variants)

- Use multilingual ASR models with Swedish support and domain adaptation when available.
- Add fuzzy name normalization (Levenshtein/phonetic matching) against known identities.
- Maintain alias dictionary: `Alex` ↔ `Alexander`, transliterations, common misspellings.

### 7) “My name is …” vs quoting/joking (false self-identification)

- Require contextual cues before accepting self-identification:
  - first-person statement near segment start,
  - repetition across turns,
  - no quotation markers/reporting verbs.
- Mark uncertain cases as `claimed_name` pending confirmation.

### 8) Mentioning others vs self-intro not distinguished reliably

- Store two separate channels:
  - `self_name_claim` (speaker about self),
  - `mentioned_name` (speaker about others).
- Never auto-enroll from `mentioned_name` alone.

### 9) Pronouns / references without names

- Add short-term coreference memory over the last N turns.
- Resolve pronouns only when one candidate is strongly dominant; otherwise leave unresolved.

### 10) Face track fragmentation (same person becomes multiple track IDs after occlusion/cuts)

- Run tracklet stitching after tracking using embedding + appearance + temporal gap constraints.
- Allow re-linking if two tracklets never overlap and are highly similar.

### 11) ID switches during tracking

- Add tracker sanity checks using motion continuity and re-ID embedding drift.
- If sudden identity jump occurs, split track and reassign segments.

### 12) Same face appears in different shots/angles/lighting (embedding mismatch)

- Use quality-weighted multi-vector identity templates (frontal/profile/lighting diversity).
- Compare against centroid plus nearest vectors, not single exemplar.

### 13) Lookalikes / family members (false matches)

- Raise match threshold for high-risk pairs and use `maybe` zone aggressively.
- Add second-factor cues when possible (voiceprint consistency, clothing continuity).

### 14) Short/blurred/profile faces (low-quality embeddings)

- Gate enrollment and matching by face quality score (size, sharpness, pose, occlusion).
- Defer identity decision until enough quality frames accumulate.

### 15) Rapid scene cuts / group shots (association timing breaks)

- Use windowed association with cut detection and reset candidate priors at hard cuts.
- In group shots, lower confidence unless speaking-face signal confirms.

### 16) Wrong label persistence (one bad enrollment causes repeated wrong labeling later)

- Enforce confirmed enrollment workflow:
  - keep auto labels provisional,
  - require human confirmation before permanent identity-store write.
- Support rollback/audit trail and vector-level deletion.

### 17) Threshold tuning issues (matched/maybe/unknown flips too often)

- Calibrate thresholds on a validation set (DET/ROC style analysis).
- Add hysteresis: higher threshold to switch identity than to keep current identity.

### 18) Multiple languages / code-switching

- Use multilingual ASR and language-ID per segment.
- Apply language-specific name parsing patterns and normalization rules.

### 19) Nicknames, diminutives, honorifics

- Store canonical identity with alias list and title stripping (`Dr.`, `Mr.`, `Ms.`).
- Resolve nickname to canonical identity only with supporting context.

### 20) Two people share same name (identity collision)

- Never key identity by name alone; key by stable `person_id`.
- Present disambiguation in UI (`name + track + confidence + thumbnail`).

### 21) Temporal mismatch (name said before/after person appears)

- Use a temporal buffer (e.g., ±5–10s) for name-to-face association.
- Decay confidence as distance from mention grows.

### 22) UI confusion (stale labels/bboxes not cleared, multiple tags overlap)

- Clear overlays each frame and expire stale labels by timestamp.
- Add collision-aware label placement and priority ordering.
- Show confidence badges and explicit `Unknown`/`Off-screen` states.

## Suggested implementation order (highest ROI first)

1. **Safety rails**: off-screen state, ambiguous state, provisional labels.
2. **Enrollment hardening**: confirmation + rollback before permanent writes.
3. **Association upgrade**: weighted multi-signal scoring + temporal buffer.
4. **Tracking robustness**: tracklet stitching and ID-switch repair.
5. **Language/name robustness**: multilingual ASR tuning + aliases/fuzzy matching.
6. **Calibration**: threshold validation and hysteresis.
7. **UI clarity**: explicit uncertainty states and overlay cleanup.

## What to monitor in production

- Name-to-face precision/recall by confidence bucket.
- Off-screen rate and ambiguous rate.
- ID switch count per hour of video.
- Enrollment reversal rate (how often humans undo automatic enrollments).
- Per-language ASR word error rate for names.

These metrics should drive threshold updates and model iteration cadence.
