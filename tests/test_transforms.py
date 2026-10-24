"""Round-trip and edge-case tests for rigid transforms."""

import math

import pytest

from egopersist.geometry.transforms import (
    Rigid,
    backproject,
    inverse_transform,
    project,
    quat_conjugate,
    quat_mul,
    quat_normalize,
    quat_rotate,
    rotation_angle_deg,
    transform_point,
)


def test_transform_point_round_trip():
    t = (1.0, -2.0, 3.0)
    q = quat_normalize((0.9, 0.2, -0.4, 0.3))
    p_cam = (0.4, -0.7, 2.1)
    # camera -> world, then world -> camera
    p_world = transform_point(t, q, p_cam)
    t_back, q_back = inverse_transform(t, q)
    p_back = transform_point(t_back, q_back, p_world)
    assert all(abs(a - b) < 1e-12 for a, b in zip(p_cam, p_back, strict=False))
    # world origin expressed in the camera frame is -R^-1 t
    origin_in_cam = transform_point(t_back, q_back, (0.0, 0.0, 0.0))
    expected = tuple(-v for v in quat_rotate(quat_conjugate(q), t))
    assert all(
        abs(a - b) < 1e-12 for a, b in zip(origin_in_cam, expected, strict=False)
    )


def test_identity_transform():
    t = (0.5, 0.0, 0.0)
    p = transform_point(t, (1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert p == (1.5, 2.0, 3.0)


def test_inverse_transform_round_trip():
    t = (1.0, -2.0, 3.0)
    q = quat_normalize((1.0, 2.0, -3.0, 0.5))
    t_cam, q_cam = inverse_transform(t, q)
    t2, q2 = inverse_transform(t_cam, q_cam)
    assert all(abs(a - b) < 1e-12 for a, b in zip(t, t2, strict=False))
    dot = abs(sum(a * b for a, b in zip(q, q2, strict=False)))
    assert dot > 1.0 - 1e-12


def test_quat_rotate_90_about_z():
    q = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    r = quat_rotate(q, (1.0, 0.0, 0.0))
    assert abs(r[0]) < 1e-12 and abs(r[1] - 1.0) < 1e-12 and abs(r[2]) < 1e-12


def test_quat_rotate_rejects_non_unit():
    with pytest.raises(ValueError):
        quat_rotate((10.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_quat_normalize_rejects_zero():
    with pytest.raises(ValueError):
        quat_normalize((0.0, 0.0, 0.0, 0.0))


def test_quat_mul_conjugate_is_rotation():
    q = quat_normalize((0.7, 0.3, -0.5, 0.5))
    r = quat_mul(q, quat_conjugate(q))
    assert abs(r[0] - 1.0) < 1e-12


def test_project_and_backproject_round_trip():
    pin = (1221.0, 1221.0, 960.0, 540.0)
    p_cam = (0.5, -0.25, 2.0)
    u, v = project(p_cam, pin)
    p_back = backproject(u, v, 2.0, pin)
    assert all(abs(a - b) < 1e-9 for a, b in zip(p_cam, p_back, strict=False))


def test_project_rejects_behind_camera():
    pin = (100.0, 100.0, 50.0, 50.0)
    with pytest.raises(ValueError):
        project((1.0, 0.0, -0.5), pin)
    with pytest.raises(ValueError):
        project((1.0, 0.0, 0.0), pin)


def test_backproject_rejects_invalid_depth():
    pin = (100.0, 100.0, 50.0, 50.0)
    with pytest.raises(ValueError):
        backproject(60.0, 40.0, 0.0, pin)
    with pytest.raises(ValueError):
        backproject(60.0, 40.0, -1.0, pin)


def test_rotation_angle_deg():
    q0 = (1.0, 0.0, 0.0, 0.0)
    q90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    assert abs(rotation_angle_deg(q0, q0)) < 1e-9
    assert abs(rotation_angle_deg(q0, q90) - 90.0) < 1e-9
    # antipodal quaternion = same rotation
    assert abs(rotation_angle_deg(q90, (-q90[0], -q90[1], -q90[2], -q90[3]))) < 1e-9


def test_rigid_compose_and_inverse():
    a = Rigid((1.0, 0.0, 0.0), quat_normalize((1.0, 0.0, 0.0, 0.5)))
    b = Rigid((2.0, 0.0, 0.0), quat_normalize((1.0, 0.0, 0.0, -0.3)))
    ab = a.compose(b)
    # ab maps b-frame points to a-frame points
    p_b = (0.3, -0.4, 2.0)
    p_a_direct = a.inverse().apply(b.apply(p_b))
    p_a_via = ab.apply(p_b)
    assert all(abs(x - y) < 1e-12 for x, y in zip(p_a_direct, p_a_via, strict=False))
    # a and b agree on the world point they both map to
    p_w = b.apply(p_b)
    assert all(
        abs(x - y) < 1e-12 for x, y in zip(a.apply(p_a_direct), p_w, strict=False)
    )


def test_rigid_inverse_apply():
    cam = Rigid((0.0, 0.0, 1.5), quat_normalize((1.0, 0.0, 0.0, 0.2)))
    p_world = (1.0, -1.0, 2.5)
    p_cam = cam.inverse().apply(p_world)
    p_back = cam.apply(p_cam)
    assert all(abs(a - b) < 1e-12 for a, b in zip(p_world, p_back, strict=False))
