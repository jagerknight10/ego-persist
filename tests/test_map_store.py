"""Tests for persistent map transactions."""

from __future__ import annotations

from wise_egomap.contracts import LabelClass, Observation
from wise_egomap.mapping import MapEntry, PersistentMap


def observation(position: tuple[float, float, float]) -> Observation:
    return Observation(
        sequence_id="seq",
        frame_id="frame",
        timestamp_s=1.0,
        instance_id="mug",
        mask_area_px=10,
        depth_support=1.0,
        co_visibility=1.0,
        appearance_similarity=1.0,
        centroid_world_m=position,
        extent_world_m=(0.1, 0.1, 0.1),
        geometry_source="rgbd",
    )


def test_map_commit_and_revert_preserve_history() -> None:
    initial = observation((0.0, 0.0, 1.0))
    moved = observation((1.0, 0.0, 1.0))
    entry = MapEntry(
        track_id="mug",
        category="cup",
        position_m=initial.centroid_world_m,
        extent_m=initial.extent_world_m,
        covariance_m2=initial.position_covariance_m2,
    )
    store = PersistentMap()
    store.add(entry)
    entry.commit(moved, 2.0)
    assert entry.map_version == 1
    assert entry.position_m == moved.centroid_world_m
    assert entry.state == LabelClass.PERSISTENT_CHANGE
    assert entry.revert(3.0) is True
    assert entry.position_m == initial.centroid_world_m
    assert entry.map_version == 2
