"""World-to-pixel projection and depth back-projection.

A ``Rigid`` pose maps camera coordinates to world coordinates, per the
project convention. ``camera.inverse()`` therefore maps world to
camera. Camera intrinsics are pinhole ``(fx, fy, cx, cy)``.
"""

from __future__ import annotations

import math

from egopersist.geometry.transforms import Rigid, backproject, project

Pinhole = tuple[float, float, float, float]


def project_world_to_pixel(
    p_world_m: tuple[float, float, float],
    camera: Rigid,
    pinhole: Pinhole,
) -> tuple[float, float]:
    """Project a world point to pixels. Raises if behind the camera."""
    p_camera = camera.inverse().apply(p_world_m)
    return project(p_camera, pinhole)


def backproject_world(
    u_px: float,
    v_px: float,
    depth_m: float,
    camera: Rigid,
    pinhole: Pinhole,
) -> tuple[float, float, float]:
    """Back-project one pixel at a known depth to a world point.

    Raises ``ValueError`` when the depth is not positive.
    """
    p_camera = backproject(u_px, v_px, depth_m, pinhole)
    return camera.apply(p_camera)


def camera_ray(
    u_px: float, v_px: float, pinhole: Pinhole
) -> tuple[float, float, float]:
    """Unit ray direction, in camera coordinates, through a pixel."""
    fx, fy, cx, cy = pinhole
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("focal lengths must be positive")
    ray = ((u_px - cx) / fx, (v_px - cy) / fy, 1.0)
    norm = math.sqrt(sum(x * x for x in ray))
    return tuple(x / norm for x in ray)  # type: ignore[return-value]


def ray_point_distance_m(
    p_world_m: tuple[float, float, float],
    camera: Rigid,
    ray_camera: tuple[float, float, float],
) -> float:
    """Minimum distance from a world point to a ray from the camera.

    ``ray_camera`` is a camera-frame direction (from :func:`camera_ray`).
    Used to attribute a 2D box observation to a 3D track: the track
    centroid must lie near the box-centre ray.
    """
    origin = camera.t_m  # camera centre in world
    ray = (
        camera.apply(
            (
                ray_camera[0] + origin[0],
                ray_camera[1] + origin[1],
                ray_camera[2] + origin[2],
            )
        )
        if False
        else camera.apply((ray_camera[0], ray_camera[1], ray_camera[2]))
    )
    # apply() translates; for a pure direction subtract the origin back
    ray = tuple(ray[i] - origin[i] for i in range(3))
    norm = math.sqrt(sum(x * x for x in ray))
    ray = tuple(x / norm for x in ray)  # type: ignore[return-value]
    p = (
        p_world_m[0] - origin[0],
        p_world_m[1] - origin[1],
        p_world_m[2] - origin[2],
    )
    t = sum(p[i] * ray[i] for i in range(3))
    if t < 0.0:
        return math.dist(p_world_m, origin)
    closest = tuple(origin[i] + t * ray[i] for i in range(3))
    return math.dist(p_world_m, closest)
