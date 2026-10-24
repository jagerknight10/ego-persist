"""Tests for the ADT reader against the local seq133 sample."""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_egomap.data import load_adt_sequence

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
