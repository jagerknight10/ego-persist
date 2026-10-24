# EgoPersist — Status and Action Plan

Date: 2026-10-24. Branch: `main`. Owner: lead agent.

## 1. Task

EgoPersist adds a causal map-maintenance layer to egocentric 4D mapping.

- Input: per-frame 2D detections, camera poses, calibration, and (on the
  cluster) depth and RGB.
- Unit of processing: one 3D instance track (e.g. one mug, one person).
- Output per track per keyframe:
  - One state: `STABLE`, `TRANSIENT`, `PERSISTENT_CHANGE`, `UNKNOWN`.
  - One map action: `keep`, `reject`, `commit`, `defer`, `revert`.
- Rule: decisions use only past and present frames (causal). A missing
  observation never removes a map element by itself.
- Demo must answer a query: "Where is the mug?" -> state, position,
  confidence, and the reason.

## 2. Dataset (verified 2026-10-24)

Dataset: ADT (Aria Digital Twin), Meta. Access file:
`Aria Digital Twin Dataset URLs.json` (236 sequences, links valid to ~2029).

Verified working: downloaded 5 small files of sequence
`Apartment_release_recognition_seq133_M1292` (~30 MB), SHA-1 matches
manifest.

Files per sequence that EgoPersist uses:

| Asset | Size/seq | Where | Used for |
|---|---|---|---|
| `main_groundtruth` (134k rows `scene_objects.csv`: per-frame object pose; `2d_bounding_box.csv`: per-frame 2D box + visibility; `3d_bounding_box.csv`: object extents; `instances.json`: identity, category, motion type; `aria_trajectory.csv`: per-frame camera pose) | ~45 MB | local + cluster | tracks, poses, 2D boxes |
| `mps_slam_calibration` (intrinsics JSONL) | ~0.2 MB | local + cluster | back-projection |
| `mps_slam_trajectories` | ~13 MB | local + cluster | camera poses (MPS) |
| `depth` | ~3.2 GB | cluster only | depth support |
| `segmentation` | ~190 MB | cluster only | mask area, appearance |
| `video_main_rgb` | 46-110 MB | local (small seqs) + cluster | appearance crops, video render |

Rules in force:

- Nothing > 100 MB per file in Git. Raw data never in Git.
- Local machine: only files < 100 MB (small test sequence).
- All bulk downloads happen on the university compute cluster.
- No synthetic training data. Local tests use the real seq133 files.
  The demo's 3 scripted scenarios (passing person, durable move, occlusion)
  are scripted decision replays for the UI, not a dataset. They are labeled
  as demo scenarios in the output.

## 3. Architecture

Six modules. Merge order: data -> geometry -> evidence -> evaluation ->
demo -> infra.

```
ADT files (cluster)
  |
  v
data:      reader + 2 Hz keyframes + rule labels + splits + audit
  |  Observation records (frozen contracts)
  v
geometry:  transforms, back-projection, 3D IoU, gated Hungarian
           association, causal state machine -> MapUpdate
  |
  v
evidence:  20-d feature vector, Bayes log-odds accumulator,
           EvidenceGRU (2-layer, calibrated) -> Decision (probabilities)
  |
  v
evaluation: event + frame metrics, machine-readable report
  |
  v
demo:      CLI replay, 3 scenarios, mug query, HTML viewer, video
infra:     Slurm jobs, Apptainer image, CI, runbook
```

Key rule: ground-truth object pose (`scene_objects.csv`) generates LABELS
only. It is never a model feature.

Model: small GRU trained on rule-generated labels. False-commit weighted
3x. Temperature calibration on validation data only. This is the only
thing that trains.

## 4. Current state

| Module | Branch | State |
|---|---|---|
| contracts | main | done, frozen |
| data | feat/data | building now (real ADT reader) |
| geometry | feat/geometry | not started (next) |
| evidence | feat/evidence | not started |
| evaluation | feat/evaluation | worker code exists, needs review |
| demo | feat/demo | worker code exists, needs review |
| infra | feat/infra | worker code exists, needs review |

Parallel `codex exec` workers failed (token burn, no output). They are
killed. The lead implements all modules directly.

Local test asset: seq133 small files in `data/raw/adt/seq/` (~30 MB,
checksum-verified). One Lite recognition sequence (video ~46 MB) will be
added for RGB-based appearance features.

## 5. Action plan (gates)

Gate G1 - data module (now)
- ADT manifest + sequence reader (real CSV/JSONL formats).
- 2 Hz deterministic keyframe sampling.
- Observation builder from 2D boxes + trajectories + extents.
- Rule-label engine from `configs/labels_v1.yaml`.
- Split validation, audit report, data card.
- PASS: pytest green on seq133; audit JSON valid.

Gate G2 - geometry module
- Transforms, back-projection, IoU, gated association, provisional
  tracks, causal state machine (commit/defer/reject/revert gates).
- PASS: pytest green (round-trips, gates, occlusion defer, reversion).

Gate G3 - evidence module
- 20-d vector, scaler, Bayes accumulator, temporal-majority baseline,
  EvidenceGRU + train/infer CLIs (lazy torch), temperature calibration.
- PASS: pytest green on CPU.

Gate G4 - merge evaluation, demo, infra
- Review worker code, fix, test, merge in order. Full suite green on
  main. Live demo run: 3 scenarios + mug query.
- PASS: `pytest` + `ruff` green on main; demo output correct.

Gate G5 - cluster package
- Download script (selected sequences + depth + segmentation + video).
- Apptainer build, 4 Slurm stages, runbook with exact commands.
- PASS: you run it, it works, results return here.

After G5 you decide: full training run or stop at the demo.

## 6. Doing now

1. This document. Commit.
2. `data` module in `.worktrees/data`, tests against seq133.
3. Pick the local RGB test sequence (video < 100 MB). Download it.
4. `geometry` module.
5. `evidence` module.
6. Review + merge evaluation, demo, infra.
7. Integration + local demo.
8. Cluster runbook + commands for you.

## 7. Risks and open questions

- ADT links are signed and expire (~2029). Cluster download must happen
  while they are valid.
- `depth` is ~3.2 GB/sequence. Full-feature runs need ~40 sequences
  minimum = ~130 GB on the cluster. Confirm disk quota.
- Camera pose source: `aria_trajectory.csv` (GT) vs MPS trajectories
  (real-time). Decision: use MPS for online realism, GT for audits.
  Flag if wrong.
- GPU budget: ~10 h (H100/H200). GRU training is tiny (minutes). Most of
  the budget goes to feature extraction over depth/segmentation.
