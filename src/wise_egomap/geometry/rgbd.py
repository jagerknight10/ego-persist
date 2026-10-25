"""Causal RGB-D object observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wise_egomap.contracts import Observation
from wise_egomap.geometry.transforms import Rigid, backproject


@dataclass(frozen=True, slots=True)
class PointCloudStats:
    """Robust geometry estimated from valid depth pixels, in metres."""

    points_world_m: np.ndarray
    centroid_world_m: tuple[float, float, float]
    extent_world_m: tuple[float, float, float]
    covariance_m2: tuple[float, ...]


def masked_color_signature(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a deterministic normalized RGB histogram for one mask."""
    image = np.asarray(rgb)
    valid_mask = np.asarray(mask, dtype=bool)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[:2] != valid_mask.shape:
        raise ValueError("rgb and mask shapes are incompatible")
    pixels = image[valid_mask]
    if len(pixels) == 0:
        raise ValueError("mask has no RGB pixels")
    bins = []
    for channel in range(min(3, pixels.shape[1])):
        values = np.clip(pixels[:, channel].astype(np.float64), 0.0, 255.0)
        bins.append(np.histogram(values, bins=8, range=(0.0, 256.0))[0])
    signature = np.concatenate(bins).astype(np.float64)
    norm = np.linalg.norm(signature)
    return signature / norm if norm > 0.0 else signature


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity clipped to [0, 1]."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, 0.0, 1.0))


def mask_points_world(
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_to_world: Rigid,
    pinhole: tuple[float, float, float, float],
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 20.0,
) -> np.ndarray:
    """Back-project valid masked depth pixels into world coordinates."""
    depth = np.asarray(depth_m, dtype=np.float64)
    valid_mask = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or valid_mask.shape != depth.shape:
        raise ValueError("depth_m and mask must be matching two-dimensional arrays")
    if min_depth_m <= 0.0 or max_depth_m <= min_depth_m:
        raise ValueError("depth limits are invalid")
    rows, cols = np.nonzero(
        valid_mask
        & np.isfinite(depth)
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
    )
    points = [
        camera_to_world.apply(
            backproject(float(col), float(row), float(depth[row, col]), pinhole)
        )
        for row, col in zip(rows.tolist(), cols.tolist(), strict=False)
    ]
    return np.asarray(points, dtype=np.float64).reshape((-1, 3))


def robust_point_stats(points_world_m: np.ndarray) -> PointCloudStats:
    """Return centroid, extent, and covariance for 3D points."""
    points = np.asarray(points_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_world_m must be a non-empty array with shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points_world_m contains non-finite values")
    median = np.median(points, axis=0)
    distance = np.linalg.norm(points - median, axis=1)
    keep = (
        distance <= np.quantile(distance, 0.98)
        if len(points) >= 10
        else np.ones(len(points), dtype=bool)
    )
    filtered = points[keep]
    centroid = np.mean(filtered, axis=0)
    extent = np.maximum(np.max(filtered, axis=0) - np.min(filtered, axis=0), 1e-6)
    covariance = (
        np.cov(filtered, rowvar=False, ddof=1)
        if len(filtered) > 1
        else np.zeros((3, 3))
    )
    covariance = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
    return PointCloudStats(
        points_world_m=filtered,
        centroid_world_m=tuple(float(v) for v in centroid),
        extent_world_m=tuple(float(v) for v in extent),
        covariance_m2=tuple(float(v) for v in covariance.reshape(-1)),
    )


def observation_from_rgbd(
    *,
    sequence_id: str,
    frame_id: str,
    timestamp_s: float,
    instance_id: str,
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_to_world: Rigid,
    pinhole: tuple[float, float, float, float],
    co_visibility: float,
    appearance_similarity: float,
    mask_area_px: int | None = None,
    min_depth_m: float = 0.05,
    max_depth_m: float = 20.0,
) -> Observation:
    """Build an Observation from current RGB-D evidence only."""
    points = mask_points_world(
        depth_m,
        mask,
        camera_to_world,
        pinhole,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    stats = robust_point_stats(points)
    total_mask = int(np.asarray(mask, dtype=bool).sum())
    return Observation(
        sequence_id=sequence_id,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        instance_id=instance_id,
        mask_area_px=mask_area_px if mask_area_px is not None else total_mask,
        depth_support=float(len(stats.points_world_m) / max(total_mask, 1)),
        co_visibility=co_visibility,
        appearance_similarity=appearance_similarity,
        centroid_world_m=stats.centroid_world_m,
        extent_world_m=stats.extent_world_m,
        valid_depth_points=len(stats.points_world_m),
        position_covariance_m2=stats.covariance_m2,
        geometry_source="rgbd",
    )
