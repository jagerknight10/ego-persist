"""Rigid transforms and pinhole projection.

Right-handed coordinates, metres, and quaternions in wxyz order.
Quaternions map camera coordinates to world coordinates unless the name
says otherwise. The Aria camera uses a FisheyeRadTanThinPrism model; this
module uses the pinhole approximation (first four intrinsic params),
which is accurate for the near-field objects EgoPersist tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Position3 = tuple[float, float, float]
QuaternionWXYZ = tuple[float, float, float, float]


def quat_mul(a: QuaternionWXYZ, b: QuaternionWXYZ) -> QuaternionWXYZ:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conjugate(q: QuaternionWXYZ) -> QuaternionWXYZ:
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_normalize(q: QuaternionWXYZ) -> QuaternionWXYZ:
    norm = math.sqrt(sum(v * v for v in q))
    if norm == 0.0:
        raise ValueError("zero quaternion")
    return tuple(v / norm for v in q)  # type: ignore[return-value]


def quat_rotate(q: QuaternionWXYZ, v: Position3) -> Position3:
    """Rotate vector v by unit quaternion q."""
    if abs(math.sqrt(sum(x * x for x in q)) - 1.0) > 1e-3:
        raise ValueError("quaternion is not unit-norm")
    qv: QuaternionWXYZ = (0.0, v[0], v[1], v[2])
    result = quat_mul(quat_mul(q, qv), quat_conjugate(q))
    return (result[1], result[2], result[3])


def transform_point(
    t_world_camera_m: Position3,
    q_world_camera: QuaternionWXYZ,
    p_camera_m: Position3,
) -> Position3:
    """Map a camera-frame point to the world frame."""
    rotated = quat_rotate(q_world_camera, p_camera_m)
    return (
        t_world_camera_m[0] + rotated[0],
        t_world_camera_m[1] + rotated[1],
        t_world_camera_m[2] + rotated[2],
    )


def inverse_transform(
    t_world_camera_m: Position3,
    q_world_camera: QuaternionWXYZ,
) -> tuple[Position3, QuaternionWXYZ]:
    """Camera-frame pose of the world origin (camera in world)."""
    q_conj = quat_conjugate(q_world_camera)
    t_camera_world = tuple(-v for v in quat_rotate(q_conj, t_world_camera_m))  # type: ignore[misc]
    return t_camera_world, q_conj


def backproject(
    u_px: float,
    v_px: float,
    depth_m: float,
    pinhole: tuple[float, float, float, float],
) -> Position3:
    """Pinhole back-projection of one pixel at a known depth."""
    if depth_m <= 0.0:
        raise ValueError("depth must be positive")
    fx, fy, cx, cy = pinhole
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("focal lengths must be positive")
    return (
        (u_px - cx) * depth_m / fx,
        (v_px - cy) * depth_m / fy,
        depth_m,
    )


def project(
    p_camera_m: Position3,
    pinhole: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Pinhole projection of a camera-frame point."""
    z = p_camera_m[2]
    if z <= 0.0:
        raise ValueError("point is not in front of the camera")
    fx, fy, cx, cy = pinhole
    return (fx * p_camera_m[0] / z + cx, fy * p_camera_m[1] / z + cy)


@dataclass(frozen=True)
class Rigid:
    """One rigid pose: translation in metres, rotation in wxyz."""

    t_m: Position3
    q_wxyz: QuaternionWXYZ

    def apply(self, p: Position3) -> Position3:
        return transform_point(self.t_m, self.q_wxyz, p)

    def inverse(self) -> "Rigid":
        t, q = inverse_transform(self.t_m, self.q_wxyz)
        return Rigid(t, q)

    def compose(self, other: "Rigid") -> "Rigid":
        """Map points in ``other``'s frame into self's frame.

        self and other must both be expressed in the same parent frame.
        """
        q = quat_mul(quat_conjugate(self.q_wxyz), other.q_wxyz)
        t = self.inverse().apply(other.t_m)
        return Rigid(t, q)


def rotation_angle_deg(a: QuaternionWXYZ, b: QuaternionWXYZ) -> float:
    """Angle in degrees between two quaternions (0..180)."""
    import math

    def _n(q):
        return quat_normalize(q)

    qa, qb = _n(a), _n(b)
    dot = abs(sum(x * y for x, y in zip(qa, qb)))
    dot = min(1.0, dot)
    return 2.0 * math.degrees(math.acos(dot))
