"""Tests for the ADT metadata audit."""

from __future__ import annotations

import json

from scripts.audit_adt_metadata import audit_sequence, main


def test_audit_reports_object_without_pose(tmp_path) -> None:
    groundtruth = tmp_path / "main_groundtruth"
    groundtruth.mkdir()
    (groundtruth / "instances.json").write_text(
        json.dumps(
            {
                "one": {"instance_id": 1},
                "two": {"instance_id": 2},
            }
        )
    )
    (groundtruth / "2d_bounding_box.csv").write_text("object_uid\n1\n2\n")
    (groundtruth / "3d_bounding_box.csv").write_text("object_uid\n1\n2\n")
    (groundtruth / "scene_objects.csv").write_text("object_uid\n1\n")
    report = audit_sequence(tmp_path)
    assert report["ids"]["instances_without_pose"] == [2]
    assert report["ids"]["2d_without_pose"] == [2]
    assert main([str(tmp_path), "--strict"]) == 1
