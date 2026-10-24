"""Oriented bounding boxes in 3D: IoU and centroid distance.

Intersection is exact. A box's six faces define six inward half-spaces.
The intersection of two boxes is a convex polytope whose vertices are:
corners of either box that lie inside the other, plus edge-and-face
crossings of either box against the other's planes that lie inside the
clipping box. The convex hull of that vertex set is the intersection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from wise_egomap.geometry.transforms import quat_normalize, quat_rotate

Position3 = tuple[float, float, float]
QuaternionWXYZ = tuple[float, float, float, float]

_EPS = 1e-9


@dataclass(frozen=True)
class OBB:
    """Centred oriented box: centroid, half-extents, and world orientation."""

    centroid_m: Position3
    half_extents_m: Position3
    orientation_wxyz: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for axis, value in enumerate(self.half_extents_m):
            if value <= 0.0:
                raise ValueError(f"half-extent {axis} must be positive")

    def corners(self) -> list[Position3]:
        q = quat_normalize(self.orientation_wxyz)
        local = [
            (
                sx * self.half_extents_m[0],
                sy * self.half_extents_m[1],
                sz * self.half_extents_m[2],
            )
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        return [
            (
                self.centroid_m[0] + quat_rotate(q, c)[0],
                self.centroid_m[1] + quat_rotate(q, c)[1],
                self.centroid_m[2] + quat_rotate(q, c)[2],
            )
            for c in local
        ]

    def planes(self) -> list[tuple[np.ndarray, float]]:
        """Six inward half-space planes ``(normal, offset)``.

        A point p is inside when ``dot(normal, p) + offset >= 0``.
        """
        q = quat_normalize(self.orientation_wxyz)
        c = np.asarray(self.centroid_m)
        h = self.half_extents_m
        planes: list[tuple[np.ndarray, float]] = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                unit = np.zeros(3)
                unit[axis] = sign  # outward direction of this face
                n_out = np.asarray(quat_rotate(q, tuple(unit)))
                face = c + h[axis] * n_out
                n_in = -n_out
                offset = -float(n_in @ face)
                planes.append((n_in, offset))
        return planes

    def volume_m3(self) -> float:
        return (
            8.0
            * self.half_extents_m[0]
            * self.half_extents_m[1]
            * self.half_extents_m[2]
        )


def _dedupe(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    return np.unique(np.round(points, 9), axis=0)


def _box_edges(corners: np.ndarray) -> list[tuple[int, int]]:
    """Index pairs of corners joined by one box edge (differ on 1 axis)."""
    edges: list[tuple[int, int]] = []
    n = corners.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if (np.abs(corners[i] - corners[j]) > 1e-7).sum() == 1:
                edges.append((i, j))
    return edges


def _inside_planes(
    points: np.ndarray, planes: list[tuple[np.ndarray, float]]
) -> np.ndarray:
    """Keep points inside every half-space (with tolerance)."""
    if points.shape[0] == 0:
        return points
    mask = np.ones(points.shape[0], dtype=bool)
    for normal, offset in planes:
        mask &= points @ normal + offset >= -_EPS
    return points[mask]


def _edge_crossings(
    edges: list[tuple[int, int]],
    corners: np.ndarray,
    planes: list[tuple[np.ndarray, float]],
) -> np.ndarray:
    """Edge-plane crossings that lie inside all given half-spaces."""
    points: list[np.ndarray] = []
    for i, j in edges:
        for normal, offset in planes:
            di = float(corners[i] @ normal + offset)
            dj = float(corners[j] @ normal + offset)
            if (di < -_EPS and dj >= -_EPS) or (di >= -_EPS and dj < -_EPS):
                t = di / (di - dj)
                points.append(corners[i] + t * (corners[j] - corners[i]))
    if not points:
        return np.empty((0, 3))
    return _inside_planes(np.asarray(points), planes)


def _hull_volume(vertices: np.ndarray) -> float:
    if vertices.shape[0] < 4:
        return 0.0
    try:
        return max(0.0, float(ConvexHull(vertices).volume))
    except QhullError:
        return 0.0


def intersection_volume_m3(a: OBB, b: OBB) -> float:
    """Exact 3D intersection volume in metres cubed."""
    va = np.asarray(a.corners())
    vb = np.asarray(b.corners())
    pa = a.planes()
    pb = b.planes()
    parts = [
        _inside_planes(va, pb),
        _inside_planes(vb, pa),
        _edge_crossings(_box_edges(va), va, pb),
        _edge_crossings(_box_edges(vb), vb, pa),
    ]
    parts = [p for p in parts if p.shape[0] > 0]
    if not parts:
        return 0.0
    return _hull_volume(_dedupe(np.vstack(parts)))


def iou_m3(a: OBB, b: OBB) -> float:
    """3D IoU, clipped to [0, 1]."""
    inter = intersection_volume_m3(a, b)
    union = a.volume_m3() + b.volume_m3() - inter
    if union <= 0.0:
        return 0.0
    return max(0.0, min(1.0, inter / union))


def centroid_distance_m(a: OBB, b: OBB) -> float:
    return float(math.dist(a.centroid_m, b.centroid_m))
