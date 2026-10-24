# Data Card — ADT (Aria Digital Twin)

## Source

ADT is Meta's Aria Digital Twin egocentric 4D dataset. Access is through
signed download links (`Aria Digital Twin Dataset URLs.json`, 236
sequences). Links are time-limited; download on the cluster while valid.
The manifest never enters Git.

## Local test asset

- Sequence: `Apartment_release_recognition_seq133_M1292`
- Files: main_groundtruth (14.5 MB), mps_slam_calibration (0.2 MB),
  mps_slam_trajectories (12.5 MB), mps_slam_summary, mps_eye_gaze (0.1 MB).
- Layout: `data/raw/adt/seq133/<asset>/...` (git-ignored).
- SHA-1 verified against the manifest.

## Fields used

| File | Field | Use |
|---|---|---|
| `main_groundtruth/instances.json` | instance id, name, category, `motion_type`, canonical pose | track identity; `dynamic` flag |
| `main_groundtruth/scene_objects.csv` | per-frame `t_wo_*`, `q_wo_*` (t = -1 canonical for static objects) | track pose timeline (labels) |
| `main_groundtruth/3d_bounding_box.csv` | local box extents | track extent |
| `main_groundtruth/2d_bounding_box.csv` | per-stream per-frame box, `visibility_ratio[%]` | visibility, mask area |
| `main_groundtruth/aria_trajectory.csv` | per-frame device pose, quality score | camera pose, depth support proxy |
| `mps_slam_calibration/online_calibration.jsonl` | 3 cameras, FisheyeRadTanThinPrism params (pinhole = first 4) | back-projection |
| `mps_slam_trajectories/*.csv` | MPS device poses | online pose alternative |

Streams: `1201-1` and `1201-2` are the RGB pair; `214-1` is the SLAM
camera. Co-visibility = fraction of streams with a box at the keyframe.

## Key rules

1. Keyframes: sequence start + integer multiples of 0.5 s (2 Hz).
2. Pose at keyframe t = last recorded pose with timestamp <= t (causal).
3. Static objects have one canonical pose row (t = -1): treated as a
   constant pose.
4. `motion_type = dynamic` -> `TRANSIENT` track. Static objects that move
   -> `PERSISTENT_CHANGE` after the confirmation hold.
5. Ground-truth pose is a LABEL input only. Never a model feature.
6. Depth support is proxied by camera trajectory quality until the
   cluster stage computes it from the `depth` asset (3.2 GB/sequence).
7. Appearance similarity is 1.0 locally; the cluster stage computes it
   from RGB crops.

## Label thresholds

See `configs/labels_v1.yaml` (change 0.10 m / 15 deg; persistent hold 5 s
over >= 5 keyframes, mode tolerance 0.05 m / 10 deg; reversion tolerance
0.05 m / 10 deg).

## Splits

Sequence-level. One sequence in exactly one split. The split file lives in
the data root on the cluster (`splits/adt_splits_v1.json`), not in Git.
