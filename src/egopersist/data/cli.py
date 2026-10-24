"""Data stage CLI.

Cluster entry point (``EGOPERSIST_PREPROCESS_MODULE``). Stage
``preprocess`` reads ADT sequences, builds keyframe observations,
generates rule labels, validates splits, and writes an audit report.

Inputs come from environment variables:
    EGOPERSIST_DATA_ROOT, EGOPERSIST_DATA_CONFIG, EGOPERSIST_LABEL_CONFIG,
    EGOPERSIST_SPLIT_FILE, EGOPERSIST_RUN_ID, EGOPERSIST_RUN_ROOT,
    EGOPERSIST_SEED.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from egopersist.contracts import EventLabel
from egopersist.data.adt import load_adt_sequence
from egopersist.data.audit import build_audit, write_audit
from egopersist.data.keyframes import build_observations, keyframe_times
from egopersist.data.label_rules import TrackSample, label_track, load_label_config
from egopersist.data.splits import load_splits, validate_splits


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"environment variable {name} is not set")
    return value


def _load_data_config(path: str) -> dict:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("version") != 1:
        raise SystemExit("unsupported data config version")
    return raw


def run_preprocess(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="egopersist.data.cli")
    parser.add_argument("--stage", required=True, choices=["preprocess"])
    args = parser.parse_args(list(argv))

    data_root = Path(_env("EGOPERSIST_DATA_ROOT"))
    data_config = _load_data_config(_env("EGOPERSIST_DATA_CONFIG"))
    label_config = load_label_config(_env("EGOPERSIST_LABEL_CONFIG"))
    run_root = Path(_env("EGOPERSIST_RUN_ROOT"))
    run_id = _env("EGOPERSIST_RUN_ID")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    split_path = Path(_env("EGOPERSIST_SPLIT_FILE"))
    splits = load_splits(split_path)
    available = sorted(
        path.name for path in data_root.iterdir() if path.is_dir()
    )
    validate_splits(splits, available)
    train_ids = set(splits["train"].sequence_ids)
    val_ids = set(splits["val"].sequence_ids)

    hz = float(data_config.get("keyframe_hz", label_config.keyframe_hz))
    all_events: list[EventLabel] = []
    audit_rows: list[dict] = []

    for sequence_id in sorted(set(available) & (train_ids | val_ids)):
        sequence = load_adt_sequence(data_root / sequence_id)
        observations = build_observations(sequence, hz)
        times = keyframe_times(sequence.start_s, sequence.end_s, hz)
        sequence_dir = run_dir / "preprocess" / sequence_id
        sequence_dir.mkdir(parents=True, exist_ok=True)

        obs_lines: list[str] = []
        label_lines: list[str] = []
        per_track_obs: dict[str, int] = {}
        n_visible = 0
        n_total = 0
        for object_uid, track_obs in sorted(observations.items()):
            instance = sequence.instances[object_uid]
            track_id = str(object_uid)
            per_track_obs[track_id] = len(track_obs)
            samples: list[TrackSample] = []
            for observation in track_obs:
                n_visible += 1
                n_total += 1
                samples.append(
                    TrackSample(
                        timestamp_s=observation.timestamp_s,
                        pose=observation.centroid_world_m,
                        orientation_wxyz=observation.orientation_wxyz,
                        visible=True,
                        dynamic=instance.motion_type == "dynamic",
                    )
                )
            events, _ = label_track(
                sequence_id, track_id, samples, label_config
            )
            all_events.extend(events)
            for event in events:
                label_lines.append(event.to_json())
            for observation in track_obs:
                obs_lines.append(observation.to_json())

        (sequence_dir / "observations.jsonl").write_text(
            "\n".join(obs_lines) + ("\n" if obs_lines else "")
        )
        (sequence_dir / "event_labels.jsonl").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else "")
        )
        n_tracks = len(observations)
        n_objects = len(sequence.instances)
        audit = build_audit(
            sequence_id,
            n_objects=n_objects,
            n_tracks=n_tracks,
            keyframe_times_s=times,
            label_config_path=_env("EGOPERSIST_LABEL_CONFIG"),
            observations_per_track=per_track_obs,
            events=[e for e in all_events if e.sequence_id == sequence_id],
            visible_fraction=(n_visible / n_total) if n_total else 0.0,
            extra={"split": "train" if sequence_id in train_ids else "val"},
        )
        audit_rows.append(audit)

    (run_dir / "preprocess" / "audit.json").write_text(
        json.dumps({"schema": "egopersist.data.audit_set/1", "audits": audit_rows}, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "preprocess" / "split_manifest.json").write_text(
        json.dumps({name: list(spec.sequence_ids) for name, spec in sorted(splits.items())}, indent=2) + "\n"
    )
    total_events = sum(sum(audit["event_counts"].values()) for audit in audit_rows)
    print(
        json.dumps(
            {
                "stage": "preprocess",
                "n_sequences": len(audit_rows),
                "n_events": total_events,
                "n_observations": sum(a["n_observations"] for a in audit_rows),
            }
        )
    )
    return 0


def main() -> int:
    return run_preprocess(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
