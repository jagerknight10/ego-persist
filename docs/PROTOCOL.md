# WISE-EgoMap Protocol v1

Status: Draft. Freeze this protocol after the ADT sample audit.

## 1. Unit of Prediction

The unit is one 3D instance track at each causal keyframe.

Each observation contains:

- an instance mask;
- metric depth;
- camera intrinsics;
- a camera-to-world pose;
- an appearance embedding;
- a timestamp.

The system must not use future frames.

## 2. Labels

| Label | Rule | Map action |
|---|---|---|
| `STABLE` | Pose agrees with the committed state | Keep |
| `TRANSIENT` | Object returns within 15 s, or a human-only region causes the change | Reject |
| `PERSISTENT_CHANGE` | New mode holds for 5 s and appears in 5 keyframes | Commit |
| `UNKNOWN` | Low visibility, low support, or heavy occlusion | Defer |

The thresholds are in `configs/labels_v1.yaml`. Freeze them before the test run.

## 3. Input Protocols

| Protocol | Masks | Depth | Pose | Purpose |
|---|---|---|---|---|
| Oracle | Ground truth | Ground truth | Ground truth | Main scientific result |
| Perception | Frozen SAM 2 | Ground truth | Ground truth | Mask robustness |
| Realistic | Frozen SAM 2 | MPS or SLAM | MPS or SLAM | Demo and gap analysis |

Ground-truth object pose is a label. It is never an input feature.

## 4. Splits

- development: 8 to 10 short sequences;
- training: 50 to 70 sequences;
- validation: 12 to 15 sequences;
- in-domain test: 15 to 20 apartment sequences;
- cross-space test: 15 to 20 office sequences;
- stress test: high-occlusion sequences.

Split by sequence and activity instance. Never split random windows from one
sequence.

## 5. Baselines

1. Never update.
2. Instant geometric update.
3. Instant semantic update.
4. Temporal majority vote.
5. Bayesian accumulator.
6. EvidenceGRU.

Lost & Found is a qualitative comparison only.

## 6. Metrics

- persistent-change precision, recall, and F1;
- transient false-commit rate;
- occlusion false-removal rate;
- time to correct commit;
- stale-map duration;
- final object position error;
- update utility at 0.30 m;
- inference latency and memory growth.

The headline plot is false-commit rate against stale-map duration.

## 7. Gates

### Data gate

Each test split must have at least 50 events in each class:

- persistent;
- transient;
- unknown or occlusion.

### Model gate

EvidenceGRU must lower the false-commit rate against instant update. The
stale-map duration must stay equal or better.

### Demo gate

The demo must show these cases:

1. A person passes and the map does not corrupt.
2. A mug moves and remains, and the map commits.
3. An occluded object does not get removed.
