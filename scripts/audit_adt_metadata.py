"""Audit ADT metadata ID and pose coverage without opening VRS files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _csv_ids(path: Path) -> tuple[set[int], int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["object_uid"]) for row in rows}, len(rows)


def audit_sequence(root: Path) -> dict[str, Any]:
    groundtruth = root / "main_groundtruth"
    instances_raw = json.loads((groundtruth / "instances.json").read_text())
    instance_ids = {int(item["instance_id"]) for item in instances_raw.values()}
    box_ids, box_rows = _csv_ids(groundtruth / "2d_bounding_box.csv")
    extent_ids, extent_rows = _csv_ids(groundtruth / "3d_bounding_box.csv")
    pose_ids, pose_rows = _csv_ids(groundtruth / "scene_objects.csv")
    return {
        "sequence": root.name,
        "counts": {
            "instances": len(instance_ids),
            "2d_box_rows": box_rows,
            "2d_box_objects": len(box_ids),
            "3d_box_rows": extent_rows,
            "3d_box_objects": len(extent_ids),
            "pose_rows": pose_rows,
            "pose_objects": len(pose_ids),
        },
        "ids": {
            "instances_without_2d": sorted(instance_ids - box_ids),
            "instances_without_3d": sorted(instance_ids - extent_ids),
            "instances_without_pose": sorted(instance_ids - pose_ids),
            "2d_without_instance": sorted(box_ids - instance_ids),
            "2d_without_pose": sorted(box_ids - pose_ids),
            "3d_without_instance": sorted(extent_ids - instance_ids),
            "3d_without_pose": sorted(extent_ids - pose_ids),
            "pose_without_instance": sorted(pose_ids - instance_ids),
        },
    }


def _has_contract_error(report: dict[str, Any]) -> bool:
    ids = report["ids"]
    return any(
        ids[name]
        for name in (
            "2d_without_instance",
            "2d_without_pose",
            "3d_without_instance",
            "3d_without_pose",
            "pose_without_instance",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_root", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    report = audit_sequence(args.sequence_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and _has_contract_error(report) else 0


if __name__ == "__main__":
    sys.exit(main())
