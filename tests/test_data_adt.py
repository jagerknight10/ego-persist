"""Tests for the ADT reader against the local seq133 sample."""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_egomap.data import load_adt_sequence
from wise_egomap.data.adt import ground_truth_pose_at

SEQ133 = Path(__file__).resolve().parents[1] / "data" / "raw" / "adt" / "seq133"


@pytest.mark.skipif(
    not (SEQ133 / "main_groundtruth").is_dir(), reason="seq133 sample not present"
)
def test_load_sequence_fields() -> None:
    seq = load_adt_sequence(SEQ133)
    assert seq.sequence_id == "seq133"
    assert seq.duration_s > 60.0
    assert len(seq.instances) == 353
    assert len(seq.extents) == 353
    assert seq.calibrations
    assert "camera-rgb" in seq.calibrations
    assert len(seq.camera_poses) > 100
    assert seq.estimated_camera_poses is not None
    assert len(seq.estimated_camera_poses) > 100
    estimated_times = [pose.tracking_timestamp_s for pose in seq.estimated_camera_poses]
    assert estimated_times == sorted(estimated_times)
    # camera poses are sorted in time
    times = [pose.tracking_timestamp_s for pose in seq.camera_poses]
    assert times == sorted(times)
    # rgb pinhole params are sane
    fx, fy, cx, cy = seq.calibrations["camera-rgb"].pinhole
    assert 100.0 < fx < 3000.0
    assert 100.0 < fy < 3000.0
    assert 0.0 < cx < 2000.0
    assert 0.0 < cy < 2000.0


@pytest.mark.skipif(
    not (SEQ133 / "main_groundtruth").is_dir(), reason="seq133 sample not present"
)
def test_missing_file_rejected(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    (partial / "main_groundtruth").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_adt_sequence(partial)


@pytest.mark.skipif(
    not (SEQ133 / "main_groundtruth").is_dir(), reason="seq133 sample not present"
)
def test_ground_truth_pose_helper_is_causal() -> None:
    sequence = load_adt_sequence(SEQ133)
    object_uid = next(iter(sequence.instances))
    timed = sorted(
        pose.timestamp_s
        for pose in sequence.object_poses
        if pose.object_uid == object_uid and pose.timestamp_s >= 0.0
    )
    if not timed:
        pytest.skip("selected object has no timed pose")
    position, _orientation = ground_truth_pose_at(sequence, object_uid, timed[-1])
    assert len(position) == 3
