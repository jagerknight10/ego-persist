"""Tests for oriented bounding box intersection and IoU."""

import math

import pytest

from egopersist.geometry.obb import (
    OBB,
    centroid_distance_m,
    iou_m3,
    intersection_volume_m3,
)

H = (0.1, 0.1, 0.1)


def test_identical_boxes_iou_one():
    a = OBB((0, 0, 0), H)
    assert abs(iou_m3(a, a) - 1.0) < 1e-9


def test_half_overlap_iou():
    # a: x in [-0.1, 0.1]; b: x in [0.0, 0.2]; same y, z.
    # inter = 0.2 * 0.2 * 0.2 = 0.008  (0.1 x-span overlap * 0.2 * 0.2)
    a = OBB((0, 0, 0), H)
    b = OBB((0.1, 0, 0), H)
    inter = intersection_volume_m3(a, b)
    # overlap in x is 0.1, y 0.2, z 0.2 -> 0.004
    assert abs(inter - 0.004) < 1e-9
    assert abs(iou_m3(a, b) - (0.004 / (0.008 + 0.008 - 0.004))) < 1e-9


def test_disjoint_boxes_zero():
    a = OBB((0, 0, 0), H)
    b = OBB((10, 0, 0), H)
    assert intersection_volume_m3(a, b) == 0.0
    assert iou_m3(a, b) == 0.0


def test_contained_box():
    outer = OBB((0, 0, 0), H)
    inner = OBB((0, 0, 0), (0.05, 0.05, 0.05))
    inter = intersection_volume_m3(outer, inner)
    assert abs(inter - 8 * 0.05**3) < 1e-9
    assert abs(iou_m3(outer, inner) - (8 * 0.05**3) / outer.volume_m3()) < 1e-9


def test_rotated_noncubic_iou():
    # 0.15 x 0.05 x 0.1 box vs itself rotated 90 deg about z.
    a = OBB((0, 0, 0), (0.15, 0.05, 0.1))
    q = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    b = OBB((0, 0, 0), (0.15, 0.05, 0.1), q)
    inter = intersection_volume_m3(a, b)
    # after 90 deg rot, b's local x aligns with a's y. Overlap:
    # x: a half 0.15, b half (rotated) 0.05 -> overlap width 2*0.05 = 0.1
    # y: a half 0.05, b half (rotated) 0.15 -> overlap 2*0.05 = 0.1
    # z: 0.1 both -> 0.2
    assert abs(inter - 0.1 * 0.1 * 0.2) < 1e-9
    assert abs(iou_m3(a, b) - 0.2) < 1e-9


def test_iou_symmetric_and_bounded():
    import random

    random.seed(0)
    for _ in range(60):
        a = OBB(
            (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)),
            (random.uniform(0.02, 0.2), random.uniform(0.02, 0.2), random.uniform(0.02, 0.2)),
        )
        b = OBB(
            (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)),
            (random.uniform(0.02, 0.2), random.uniform(0.02, 0.2), random.uniform(0.02, 0.2)),
        )
        assert abs(iou_m3(a, b) - iou_m3(b, a)) < 1e-9
        assert 0.0 <= iou_m3(a, b) <= 1.0 + 1e-12


def test_centroid_distance():
    a = OBB((0, 0, 0), H)
    b = OBB((3, 4, 0), H)
    assert abs(centroid_distance_m(a, b) - 5.0) < 1e-12


def test_invalid_half_extent_rejected():
    with pytest.raises(ValueError):
        OBB((0, 0, 0), (0.1, 0.0, 0.1))
