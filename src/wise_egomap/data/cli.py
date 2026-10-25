"""Data stage CLI.

Cluster entry point (``WISE_EGOMAP_PREPROCESS_MODULE``). Stage
``preprocess`` reads ADT sequences, builds keyframe observations,
generates rule labels, validates splits, and writes an audit report.

Inputs come from environment variables:
    WISE_EGOMAP_DATA_ROOT, WISE_EGOMAP_DATA_CONFIG, WISE_EGOMAP_LABEL_CONFIG,
    WISE_EGOMAP_SPLIT_FILE, WISE_EGOMAP_RUN_ID, WISE_EGOMAP_RUN_ROOT,
    WISE_EGOMAP_SEED.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from wise_egomap.contracts import EventLabel
from wise_egomap.data.adt import ground_truth_pose_at, load_adt_sequence
from wise_egomap.data.audit import build_audit
from wise_egomap.data.keyframes import (
    build_observations,
    build_rgbd_observations,
    keyframe_times,
)
from wise_egomap.data.label_rules import TrackSample, label_track, load_label_config
from wise_egomap.data.splits import load_splits, validate_splits
from wise_egomap.data.vrs import AriaVrsReader, find_vrs_file


def _depth_at(reader: AriaVrsReader, label: str, timestamp_s: float):
    """Read an ADT depth frame and convert millimetres to metres."""
    return reader.image_at(label, timestamp_s).astype("float32") / 1000.0


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
    parser = argparse.ArgumentParser(prog="wise_egomap.data.cli")
    parser.add_argument("--stage", required=True, choices=["preprocess"])
    parser.parse_args(list(argv))

    data_root = Path(_env("WISE_EGOMAP_DATA_ROOT"))
    data_config = _load_data_config(_env("WISE_EGOMAP_DATA_CONFIG"))
    label_config = load_label_config(_env("WISE_EGOMAP_LABEL_CONFIG"))
    run_root = Path(_env("WISE_EGOMAP_RUN_ROOT"))
    run_id = _env("WISE_EGOMAP_RUN_ID")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    split_path = Path(_env("WISE_EGOMAP_SPLIT_FILE"))
    splits = load_splits(split_path)
    available = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    validate_splits(splits, available)
    from wise_egomap.data.splits import SplitSpec

    train_spec = splits.get("train", SplitSpec("train", ()))
    val_spec = splits.get("val", SplitSpec("val", ()))
    if "train" not in splits:
        raise SystemExit("split manifest has no 'train' split")
    train_ids = set(train_spec.sequence_ids)
    val_ids = set(val_spec.sequence_ids)
    test_ids = set(splits.get("test", SplitSpec("test", ())).sequence_ids)
    selected_ids = train_ids | val_ids | test_ids
    missing = sorted(selected_ids - set(available))
    if missing:
        raise SystemExit(
            "split references sequences that are absent from the data root: "
            + ", ".join(missing)
        )

    hz = float(data_config.get("keyframe_hz", label_config.keyframe_hz))
    all_events: list[EventLabel] = []
    audit_rows: list[dict] = []

    geometry_mode = data_config.get("observation_geometry", "rgbd")
    for sequence_id in sorted(set(available) & selected_ids):
        sequence = load_adt_sequence(data_root / sequence_id)
        if geometry_mode == "rgbd":
            depth_dir = data_root / sequence_id / data_config["depth_asset"]
            depth_reader = AriaVrsReader(find_vrs_file(depth_dir))
            depth_label = str(data_config.get("depth_stream_label", "camera-rgb"))
            mask_provider = None
            rgb_provider = None
            if data_config.get("rgb_asset"):
                rgb_dir = data_root / sequence_id / data_config["rgb_asset"]
                rgb_reader = AriaVrsReader(find_vrs_file(rgb_dir))
                rgb_label = str(data_config.get("rgb_stream_label", "camera-rgb"))

                def rgb_provider(
                    timestamp_s: float,
                    reader=rgb_reader,
                    label=rgb_label,
                ):
                    return reader.image_at(label, timestamp_s)

            if data_config.get("mask_source") == "segmentation":
                segmentation_dir = (
                    data_root / sequence_id / data_config["segmentation_asset"]
                )
                segmentation_reader = AriaVrsReader(find_vrs_file(segmentation_dir))
                segmentation_label = str(
                    data_config.get("segmentation_stream_label", "camera-rgb")
                )

                def mask_provider(
                    timestamp_s: float,
                    object_uid: int,
                    reader=segmentation_reader,
                    label=segmentation_label,
                ):
                    return reader.image_at(label, timestamp_s) == object_uid

            observations = build_rgbd_observations(
                sequence,
                hz,
                lambda timestamp_s, reader=depth_reader, label=depth_label: _depth_at(
                    reader, label, timestamp_s
                ),
                rgb_stream=str(data_config.get("rgb_stream", "214-1")),
                pose_source=str(data_config.get("pose_source", "gt")),
                mask_provider=mask_provider,
                rgb_provider=rgb_provider,
            )
        elif geometry_mode == "legacy_gt":
            if not data_config.get("allow_legacy_gt_geometry", False):
                raise SystemExit("legacy_gt geometry is disabled by the data config")
            observations = build_observations(sequence, hz)
        else:
            raise SystemExit(f"unsupported observation_geometry: {geometry_mode}")
        times = keyframe_times(sequence.start_s, sequence.end_s, hz)
        sequence_dir = run_dir / "preprocess" / sequence_id
        sequence_dir.mkdir(parents=True, exist_ok=True)

        obs_lines: list[str] = []
        label_lines: list[str] = []
        track_label_map: dict[str, list] = {}
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
                gt_position, gt_orientation = ground_truth_pose_at(
                    sequence, object_uid, observation.timestamp_s
                )
                samples.append(
                    TrackSample(
                        timestamp_s=observation.timestamp_s,
                        pose=gt_position,
                        orientation_wxyz=gt_orientation,
                        visible=True,
                        dynamic=instance.motion_type == "dynamic",
                    )
                )
            events, per_keyframe = label_track(
                sequence_id, track_id, samples, label_config
            )
            assert len(per_keyframe) == len(track_obs)
            track_label_map[track_id] = per_keyframe
            all_events.extend(events)
            for event in events:
                label_lines.append(event.to_json())
            for observation in track_obs:
                obs_lines.append(observation.to_json())

        (sequence_dir / "observations.jsonl").write_text(
            "\n".join(obs_lines) + ("\n" if obs_lines else "")
        )
        (sequence_dir / "track_labels.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {"track_id": track_id, "labels": [lc.value for lc in labels]}
                )
                for track_id, labels in sorted(track_label_map.items())
            )
            + ("\n" if track_label_map else "")
        )
        meta_lines = [
            json.dumps(
                {
                    "track_id": str(uid),
                    "motion_type": sequence.instances[uid].motion_type,
                }
            )
            for uid in sorted(observations)
        ]
        (sequence_dir / "track_meta.jsonl").write_text(
            "\n".join(meta_lines) + ("\n" if meta_lines else "")
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
            label_config_path=_env("WISE_EGOMAP_LABEL_CONFIG"),
            observations_per_track=per_track_obs,
            events=[e for e in all_events if e.sequence_id == sequence_id],
            visible_fraction=(n_visible / n_total) if n_total else 0.0,
            extra={
                "split": (
                    "train"
                    if sequence_id in train_ids
                    else "val"
                    if sequence_id in val_ids
                    else "test"
                ),
                "geometry_source": geometry_mode,
            },
        )
        audit_rows.append(audit)

    (run_dir / "preprocess" / "audit.json").write_text(
        json.dumps(
            {"schema": "wise_egomap.data.audit_set/1", "audits": audit_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (run_dir / "preprocess" / "split_manifest.json").write_text(
        json.dumps(
            {name: list(spec.sequence_ids) for name, spec in sorted(splits.items())},
            indent=2,
        )
        + "\n"
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
