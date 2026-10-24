"""Projection and back-projection round trips."""

import math

import pytest

from wise_egomap.geometry.projection import (
    backproject_world,
    camera_ray,
    project_world_to_pixel,
    ray_point_distance_m,
)
from wise_egomap.geometry.transforms import Rigid, quat_normalize, transform_point

PIN = (1221.0, 1221.0, 960.0, 540.0)


def test_round_trip_axis_camera():
    cam = Rigid((0.0, 0.0, 1.5), (1.0, 0.0, 0.0, 0.0))
    p_world = (0.1, -0.2, 2.1)
    u, v = project_world_to_pixel(p_world, cam, PIN)
    depth = cam.inverse().apply(p_world)[2]
    p_back = backproject_world(u, v, depth, cam, PIN)
    assert all(abs(a - b) < 1e-9 for a, b in zip(p_world, p_back, strict=False))


def test_round_trip_rotated_camera():
    cam = Rigid((1.0, 1.0, 0.5), quat_normalize((1.0, 0.0, 0.0, 0.35)))
    p_cam = (1.0, -0.5, 1.2)
    p_world = transform_point(cam.t_m, cam.q_wxyz, p_cam)
    u, v = project_world_to_pixel(p_world, cam, PIN)
    p_back = backproject_world(u, v, p_cam[2], cam, PIN)
    assert all(abs(a - b) < 1e-9 for a, b in zip(p_world, p_back, strict=False))


def test_invalid_depth_rejected():
    cam = Rigid((0.0, 0.0, 1.5), (1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        backproject_world(960.0, 540.0, 0.0, cam, PIN)
    with pytest.raises(ValueError):
        backproject_world(960.0, 540.0, -0.5, cam, PIN)


def test_point_behind_camera_rejected():
    cam = Rigid((0.0, 0.0, 1.5), (1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        project_world_to_pixel((0.0, 0.0, 1.4), cam, PIN)
    with pytest.raises(ValueError):
        project_world_to_pixel((0.0, 0.0, 1.5), cam, PIN)


def test_camera_ray_unit():
    r = camera_ray(1000.0, 500.0, PIN)
    assert abs(math.sqrt(sum(x * x for x in r)) - 1.0) < 1e-12


def test_ray_point_distance():
    cam = Rigid((0.0, 0.0, 1.5), (1.0, 0.0, 0.0, 0.0))
    p_world = (0.1, -0.2, 2.1)
    u, v = project_world_to_pixel(p_world, cam, PIN)
    r = camera_ray(u, v, PIN)
    assert ray_point_distance_m(p_world, cam, r) < 1e-9
    # offset perpendicular to the ray in the image plane
    expected = 0.05 * math.sqrt(1.0 - r[0] ** 2)
    got = ray_point_distance_m((p_world[0] + 0.05, p_world[1], p_world[2]), cam, r)
    assert abs(got - expected) < 1e-9
