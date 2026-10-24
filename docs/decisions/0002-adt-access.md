# ADR 0002: ADT Access via User-Provided Link Manifest

Status: Accepted on 2026-10-25.

## Context

ADR 0001 assumed no dataset licence and set a licence gate.
`docs/ADT_ACCESS.md` therefore forbade any ADT download.

The owner then supplied `Aria Digital Twin Dataset URLs.json`
(236 sequences, signed download links valid to ~2029) obtained
through the owner's own access channel. That file is the official
Meta Aria Digital Twin download-link format.

## Decision

1. Owner-provided access supersedes the licence gate. ADT downloads
   are permitted.
2. The link manifest is treated as a credential:
   - never committed to this repository;
   - never pasted into chat;
   - on the cluster it lives at `$DATA_ROOT/manifests/adt_urls.json`.
3. Poses: `aria_trajectory.csv` (ground truth) is the camera pose
   source. Ground truth is used for labels and evaluation only.
   Features use only causal, online-available quantities.
4. Pipeline asset scope: `main_groundtruth` + `mps_slam_calibration`
   only (poses, boxes, extents, intrinsics). Video, depth, and
   segmentation are not required for the v1 pipeline.
5. Local testing is limited to assets under 100 MB (seq133, ~30 MB).
   All other data lands on the cluster.

## Consequences

- `docs/ADT_ACCESS.md` "no download until licence approved" is void;
  the owner-provided manifest is the approval.
- `scripts/download_adt_json.py` is the sanctioned downloader
  (HTTPS only, SHA-1 and size verified, safe-zip extract).
- `scripts/download_adt_subset.sh` (generic URL list) remains as a
  fallback; the runbook uses the manifest downloader.
