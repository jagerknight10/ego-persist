"""Tests for RGB-D geometry construction."""

from __future__ import annotations

import numpy as np

from wise_egomap.data.adt import (
    AdtSequence,
    Box2D,
    CameraCalibration,
    CameraPose,
    Instance,
    ObjectExtent,
)
from wise_egomap.data.keyframes import build_rgbd_observations
from wise_egomap.geometry import (
    cosine_similarity,
    mask_points_world,
    masked_color_signature,
    observation_from_rgbd,
)
from wise_egomap.geometry.transforms import Rigid


def test_mask_points_world_backprojects_current_depth() -> None:
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    mask = np.array([[True, False], [True, False]])
    points = mask_points_world(
        depth,
        mask,
        Rigid((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0)),
        (1.0, 1.0, 0.0, 0.0),
    )
    assert points.shape == (2, 3)
    assert np.allclose(points[0], (1.0, 2.0, 5.0))
    assert np.allclose(points[1], (1.0, 4.0, 5.0))


def test_rgbd_observation_marks_geometry_source_and_statistics() -> None:
    depth = np.full((3, 3), 1.0, dtype=np.float32)
    mask = np.ones((3, 3), dtype=bool)
    observation = observation_from_rgbd(
        sequence_id="seq",
        frame_id="frame",
        timestamp_s=1.0,
        instance_id="mug",
        depth_m=depth,
        mask=mask,
        camera_to_world=Rigid((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        pinhole=(1.0, 1.0, 1.0, 1.0),
        co_visibility=1.0,
        appearance_similarity=0.8,
    )
    assert observation.geometry_source == "rgbd"
    assert observation.valid_depth_points == 9
    assert observation.depth_support == 1.0
    assert len(observation.position_covariance_m2) == 9
    assert observation.centroid_world_m[2] == 1.0


def test_rgbd_observation_rejects_empty_depth_support() -> None:
    depth = np.zeros((2, 2), dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    try:
        observation_from_rgbd(
            sequence_id="seq",
            frame_id="frame",
            timestamp_s=1.0,
            instance_id="mug",
            depth_m=depth,
            mask=mask,
            camera_to_world=Rigid((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            pinhole=(1.0, 1.0, 0.0, 0.0),
            co_visibility=1.0,
            appearance_similarity=1.0,
        )
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty depth must not produce an observation")


def test_rgb_appearance_signature_is_deterministic() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    mask = np.ones((2, 2), dtype=bool)
    signature = masked_color_signature(rgb, mask)
    assert np.isclose(cosine_similarity(signature, signature), 1.0)


def test_rgbd_keyframes_do_not_read_object_pose_table() -> None:
    object_id = 7
    sequence = AdtSequence(
        sequence_id="seq",
        dataset_name="ADT",
        scene="Apartment",
        instances={
            object_id: Instance(
                instance_id=object_id,
                instance_name="mug",
                prototype_name="mug",
                category="cup",
                motion_type="static",
                rigidity="rigid",
                up_vector=(0.0, 1.0, 0.0),
                front_vector=(0.0, 0.0, 1.0),
            )
        },
        extents={object_id: ObjectExtent(object_id, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)},
        object_poses=[],
        boxes_2d=[Box2D("214-1", object_id, 0.0, 0.0, 2.0, 0.0, 2.0, 1.0)],
        camera_poses=[CameraPose(0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 1.0)],
        calibrations={
            "camera-rgb": CameraCalibration(
                "camera-rgb",
                "",
                "pinhole",
                (1.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            )
        },
        start_s=0.0,
        end_s=0.0,
    )
    observations = build_rgbd_observations(
        sequence,
        2.0,
        lambda _timestamp_s: np.ones((3, 3), dtype=np.float32),
    )
    assert observations[object_id][0].geometry_source == "rgbd"


def test_rgbd_keyframes_drop_objects_without_valid_depth() -> None:
    object_id = 7
    sequence = AdtSequence(
        sequence_id="seq",
        dataset_name="ADT",
        scene="Apartment",
        instances={
            object_id: Instance(
                instance_id=object_id,
                instance_name="mug",
                prototype_name="mug",
                category="cup",
                motion_type="static",
                rigidity="rigid",
                up_vector=(0.0, 1.0, 0.0),
                front_vector=(0.0, 0.0, 1.0),
            )
        },
        extents={object_id: ObjectExtent(object_id, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)},
        object_poses=[],
        boxes_2d=[Box2D("214-1", object_id, 0.0, 0.0, 2.0, 0.0, 2.0, 1.0)],
        camera_poses=[CameraPose(0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 1.0)],
        calibrations={
            "camera-rgb": CameraCalibration(
                "camera-rgb",
                "",
                "pinhole",
                (1.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            )
        },
        start_s=0.0,
        end_s=0.0,
    )
    observations = build_rgbd_observations(
        sequence,
        2.0,
        lambda _timestamp_s: np.zeros((3, 3), dtype=np.float32),
    )
    assert object_id not in observations
