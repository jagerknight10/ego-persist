"""Tests for split validation and the audit report."""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_egomap.contracts import EventLabel, EventType, LabelClass
from wise_egomap.data import (
    SplitSpec,
    build_audit,
    load_splits,
    validate_splits,
    write_audit,
)


def test_split_overlap_rejected() -> None:
    splits = {
        "train": SplitSpec("train", ("a", "b")),
        "val": SplitSpec("val", ("b", "c")),
    }
    with pytest.raises(ValueError, match="both"):
        validate_splits(splits, {"a", "b", "c"})


def test_unknown_sequence_rejected() -> None:
    splits = {"train": SplitSpec("train", ("a", "ghost"))}
    with pytest.raises(ValueError, match="unknown"):
        validate_splits(splits, {"a"})


def test_load_splits_roundtrip(tmp_path: Path) -> None:
    import json

    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"train": ["a", "b"], "val": ["c"], "test": ["d"]}))
    splits = load_splits(path)
    assert set(splits) == {"train", "val", "test"}
    assert splits["train"].sequence_ids == ("a", "b")
    validate_splits(splits, {"a", "b", "c", "d"})


def test_audit_report_required_fields(tmp_path: Path) -> None:
    config = tmp_path / "labels.yaml"
    config.write_text("version: 1\n")
    audit = build_audit(
        "seq133",
        n_objects=353,
        n_tracks=314,
        keyframe_times_s=[0.0, 0.5, 1.0],
        label_config_path=str(config),
        observations_per_track={"1": 2, "2": 1},
        events=[
            EventLabel(
                sequence_id="seq133",
                track_id="1",
                event_type=EventType.STABLE,
                start_s=0.0,
                end_s=1.0,
                label=LabelClass.STABLE,
            )
        ],
        visible_fraction=0.9,
    )
    for field in (
        "schema",
        "sequence_id",
        "n_objects",
        "n_tracks",
        "n_keyframes",
        "n_observations",
        "visible_fraction",
        "event_counts",
        "label_counts",
        "label_config_sha256",
    ):
        assert field in audit
    assert audit["n_observations"] == 3
    assert audit["label_counts"]["STABLE"] == 1
    write_audit(audit, tmp_path / "audit.json")
    assert (tmp_path / "audit.json").is_file()
