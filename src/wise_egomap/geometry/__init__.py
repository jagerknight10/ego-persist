"""Geometry: transforms, projection, boxes, association, and RGB-D stats."""

from wise_egomap.geometry.rgbd import (
    PointCloudStats,
    cosine_similarity,
    mask_points_world,
    masked_color_signature,
    observation_from_rgbd,
    robust_point_stats,
)

__all__ = [
    "PointCloudStats",
    "cosine_similarity",
    "mask_points_world",
    "masked_color_signature",
    "observation_from_rgbd",
    "robust_point_stats",
]
