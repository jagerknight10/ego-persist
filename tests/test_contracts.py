"""Contract tests that protect module boundaries."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from egopersist.contracts import (
    Decision,
    DecisionAction,
    EventLabel,
    EventType,
    EvidenceVector,
    LabelClass,
    MapUpdate,
    Observation,
    RunManifest,
    TrackObservation,
)


def observation() -> Observation:
    return Observation(
        sequence_id="seq-1",
        frame_id="frame-001",
        timestamp_s=1.5,
        instance_id="mug",
        mask_area_px=1200,
        depth_support=0.9,
        co_visibility=0.8,
        appearance_similarity=0.95,
        centroid_world_m=(0.1, 0.2, 0.3),
        extent_world_m=(0.1, 0.1, 0.2),
        displacement_m=0.2,
        rotation_deg=5.0,
    )


def probabilities() -> dict[LabelClass, float]:
    return {
        LabelClass.STABLE: 0.1,
        LabelClass.TRANSIENT: 0.1,
        LabelClass.PERSISTENT_CHANGE: 0.7,
        LabelClass.UNKNOWN: 0.1,
    }


def test_observation_json_round_trip() -> None:
    item = observation()
    assert Observation.from_json(item.to_json()) == item


def test_observation_rejects_zero_extent() -> None:
    with pytest.raises(ValueError, match="positive"):
        Observation(
            sequence_id="seq-1",
            frame_id="frame-001",
            timestamp_s=1.5,
            instance_id="mug",
            mask_area_px=1200,
            depth_support=0.9,
            co_visibility=0.8,
            appearance_similarity=0.95,
            centroid_world_m=(0.0, 0.0, 0.0),
            extent_world_m=(0.0, 0.1, 0.1),
        )


def test_observation_rejects_unknown_field() -> None:
    data = observation().to_dict()
    data["future_leak"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        Observation.from_dict(data)


def test_evidence_vector_shape() -> None:
    vector = EvidenceVector(feature_names=("a", "b"), values=(0.1, 0.2))
    assert vector.values == (0.1, 0.2)
    with pytest.raises(ValueError, match="equal length"):
        EvidenceVector(feature_names=("a",), values=(0.1, 0.2))


def test_decision_json_round_trip() -> None:
    item = Decision(
        sequence_id="seq-1",
        track_id="track-1",
        frame_id="frame-001",
        timestamp_s=1.5,
        label=LabelClass.PERSISTENT_CHANGE,
        action=DecisionAction.COMMIT,
        confidence=0.7,
        probabilities=probabilities(),
        reason="five visible keyframes",
    )
    restored = Decision.from_json(item.to_json())
    assert restored == item
    assert restored.probabilities[LabelClass.PERSISTENT_CHANGE] == 0.7


def test_decision_requires_normalized_probabilities() -> None:
    data = probabilities()
    data[LabelClass.STABLE] = 0.2
    with pytest.raises(ValueError, match="sum to 1.0"):
        Decision(
            sequence_id="seq-1",
            track_id="track-1",
            frame_id="frame-001",
            timestamp_s=1.5,
            label=LabelClass.PERSISTENT_CHANGE,
            action=DecisionAction.COMMIT,
            confidence=0.7,
            probabilities=data,
            reason="test",
        )


def test_map_update_requires_position_for_commit() -> None:
    with pytest.raises(ValueError, match="requires new_position_m"):
        MapUpdate(
            sequence_id="seq-1",
            track_id="track-1",
            timestamp_s=1.0,
            action=DecisionAction.COMMIT,
            map_version=1,
        )


def test_event_label_window_order() -> None:
    with pytest.raises(ValueError, match="end_s"):
        EventLabel(
            sequence_id="seq-1",
            track_id="track-1",
            event_type=EventType.PERSISTENT_CHANGE,
            start_s=5.0,
            end_s=4.0,
            label=LabelClass.PERSISTENT_CHANGE,
        )


def test_track_observation_json_round_trip() -> None:
    item = TrackObservation(
        track_id="track-1",
        observation=observation(),
        committed_position_m=(0.0, 0.0, 0.0),
    )
    assert TrackObservation.from_json(item.to_json()) == item


def test_run_manifest_json_round_trip() -> None:
    item = RunManifest(
        run_id="run-1",
        git_commit="abcdef0",
        git_dirty=False,
        command=("python", "train.py"),
        config_path="config.yaml",
        config_hash="a" * 64,
        data_split_hash="b" * 64,
        label_config_hash="c" * 64,
        seed=0,
        started_at_utc=datetime(2026, 10, 24, tzinfo=timezone.utc),
        hostname="node-1",
        python_version="3.11",
        packages={"torch": "2.3.0"},
        extra={"gpu": "A100"},
    )
    assert RunManifest.from_json(item.to_json()) == item
