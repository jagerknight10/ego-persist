"""Deterministic keyframe sampling and Observation construction.

Keyframes are the sequence start plus integer multiples of ``1 / hz``.
Sampling never looks ahead: the pose used at keyframe time t is the last
recorded pose with timestamp <= t.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from wise_egomap.contracts import Observation
from wise_egomap.data.adt import AdtSequence, Box2D, CameraCalibration

RGB_STREAM_CANDIDATES = ("214-1", "1201-1", "1201-2")


def keyframe_times(start_s: float, end_s: float, hz: float) -> list[float]:
    """Return deterministic keyframe times in [start_s, end_s]."""
    if hz <= 0.0:
        raise ValueError("hz must be positive")
    if end_s < start_s:
        raise ValueError("end_s must be >= start_s")
    step = 1.0 / hz
    count = int(math.floor((end_s - start_s) / step)) + 1
    return [start_s + index * step for index in range(count)]


def _pose_tracks(
    seq: AdtSequence,
) -> tuple[
    dict[int, list[tuple[float, tuple, tuple]]],
    dict[int, tuple[tuple, tuple]],
]:
    """Split object poses into per-frame tracks and canonical static poses.

    Objects with timed rows (t >= 0) get a track. Objects whose only row is
    the canonical pose (t = -1, static objects) get a constant pose.
    """
    tracks: dict[int, list[tuple[float, tuple, tuple]]] = defaultdict(list)
    canonical: dict[int, tuple[tuple, tuple]] = {}
    for pose in seq.object_poses:
        if pose.timestamp_s >= 0.0:
            tracks[pose.object_uid].append(
                (pose.timestamp_s, pose.t_world_m, pose.q_world_wxyz)
            )
        else:
            canonical[pose.object_uid] = (pose.t_world_m, pose.q_world_wxyz)
    for items in tracks.values():
        items.sort(key=lambda item: item[0])
    return tracks, canonical


def _box_streams(
    seq: AdtSequence,
) -> dict[str, dict[int, list[tuple[float, float, int, float]]]]:
    """Per stream, per object: (t, visibility, area_px, is_rgb)."""
    out: dict[str, dict[int, list[tuple[float, float, int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for box in seq.boxes_2d:
        out[box.stream_id][box.object_uid].append(
            (box.timestamp_s, box.visibility_ratio, box.area_px, box.stream_id)
        )
    for stream in out.values():
        for items in stream.values():
            items.sort(key=lambda item: item[0])
    return out


def _camera_quality(seq: AdtSequence) -> tuple[list[float], list[float]]:
    times = [pose.tracking_timestamp_s for pose in seq.camera_poses]
    scores = [pose.quality_score for pose in seq.camera_poses]
    return times, scores


def _quality_at(times: list[float], scores: list[float], t: float) -> float:
    if not times:
        return 0.0
    index = bisect.bisect_right(times, t) - 1
    if index < 0:
        return scores[0]
    return scores[index]


def _box_at(
    items: list[tuple[float, float, int, float]] | None, t: float
) -> tuple[float, int] | None:
    """Latest (visibility, area_px) at time t, or None."""
    if not items:
        return None
    index = bisect.bisect_right([item[0] for item in items], t) - 1
    if index < 0:
        return None
    return items[index][1], items[index][2]


def build_observations(
    seq: AdtSequence,
    hz: float,
    *,
    rgb_stream: str | None = None,
) -> dict[int, list[Observation]]:
    """Build one Observation list per object at deterministic keyframes.

    Visibility comes from 2D annotation boxes. Depth support comes from the
    camera trajectory quality score. Appearance similarity is 1.0 until
    appearance features are available on the cluster.
    """
    times = keyframe_times(seq.start_s, seq.end_s, hz)
    tracks, canonical = _pose_tracks(seq)
    streams = _box_streams(seq)
    cam_times, cam_scores = _camera_quality(seq)

    stream_ids = list(streams)
    if rgb_stream is None:
        for candidate in RGB_STREAM_CANDIDATES:
            if candidate in stream_ids:
                rgb_stream = candidate
                break
        if rgb_stream is None:
            rgb_stream = stream_ids[0] if stream_ids else None
    n_streams = len(stream_ids)

    previous: dict[int, tuple] = {}
    out: dict[int, list[Observation]] = {}
    for object_uid in sorted(seq.instances):
        extent = seq.extents.get(object_uid)
        if extent is None:
            continue
        if object_uid in tracks:
            track = tracks[object_uid]
            track_times = [item[0] for item in track]
            constant_pose: tuple | None = None
        elif object_uid in canonical:
            track = []
            track_times = []
            constant_pose = canonical[object_uid]
        else:
            continue
        rgb_items = streams.get(rgb_stream, {}).get(object_uid) if rgb_stream else None
        all_items = {stream: items.get(object_uid) for stream, items in streams.items()}
        observations: list[Observation] = []
        for t in times:
            if constant_pose is not None:
                pose, orientation = constant_pose
            else:
                index = bisect.bisect_right(track_times, t) - 1
                if index < 0:
                    continue
                _, pose, orientation = track[index]
            rgb = _box_at(rgb_items, t)
            if rgb is None:
                continue
            visibility, area_px = rgb
            co_visibility = 0.0
            if n_streams > 0:
                seen = sum(
                    1
                    for stream, items in all_items.items()
                    if _box_at(items, t) is not None
                )
                co_visibility = seen / n_streams
            depth_support = max(0.0, min(1.0, _quality_at(cam_times, cam_scores, t)))
            centroid = (float(pose[0]), float(pose[1]), float(pose[2]))
            # Appearance similarity is 1.0 until RGB crops (cluster stage)
            # replace the placeholder.
            appearance = 1.0
            prev = previous.get(object_uid)
            displacement = 0.0
            rotation = 0.0
            if prev is not None:
                prev_pose, prev_ori = prev
                displacement = math.dist(pose, prev_pose)
                dots = (a * b for a, b in zip(orientation, prev_ori, strict=False))
                dot = max(-1.0, min(1.0, abs(sum(dots))))
                rotation = math.degrees(2.0 * math.acos(dot))
            previous[object_uid] = (pose, orientation)
            observations.append(
                Observation(
                    sequence_id=seq.sequence_id,
                    frame_id=f"kf{len(observations):06d}",
                    timestamp_s=round(t, 6),
                    instance_id=str(object_uid),
                    mask_area_px=max(1, int(area_px)),
                    depth_support=round(depth_support, 6),
                    co_visibility=round(co_visibility, 6),
                    appearance_similarity=appearance,
                    centroid_world_m=centroid,
                    extent_world_m=extent.size_m,
                    orientation_wxyz=orientation,
                    displacement_m=round(displacement, 6),
                    rotation_deg=round(rotation, 6),
                )
            )
        if observations:
            out[object_uid] = observations
    return out


def build_rgbd_observations(
    seq: AdtSequence,
    hz: float,
    depth_provider: Callable[[float], np.ndarray],
    *,
    rgb_stream: str = "214-1",
    pose_source: str = "gt",
    mask_provider: Callable[[float, int], np.ndarray] | None = None,
    rgb_provider: Callable[[float], np.ndarray] | None = None,
) -> dict[int, list[Observation]]:
    """Build observations from current depth and box masks.

    The boxes provide oracle masks for the controlled protocol. The object
    pose table is never read by this function. ``pose_source`` selects camera
    poses only: ``gt`` is controlled evaluation and ``mps`` is realistic
    evaluation.
    """
    from wise_egomap.geometry.rgbd import (
        cosine_similarity,
        masked_color_signature,
        observation_from_rgbd,
    )
    from wise_egomap.geometry.transforms import Rigid, quat_mul, quat_rotate

    if pose_source not in {"gt", "mps"}:
        raise ValueError("pose_source must be 'gt' or 'mps'")
    camera_poses = (
        seq.estimated_camera_poses if pose_source == "mps" else seq.camera_poses
    )
    if not camera_poses:
        raise ValueError(f"no camera poses available for pose source {pose_source}")
    calibration = seq.calibrations.get("camera-rgb")
    if calibration is None:
        raise ValueError("camera-rgb calibration is required")
    times = keyframe_times(seq.start_s, seq.end_s, hz)
    boxes_by_object: dict[int, list[Box2D]] = defaultdict(list)
    for box in seq.boxes_2d:
        if box.stream_id == rgb_stream:
            boxes_by_object[box.object_uid].append(box)
    for boxes in boxes_by_object.values():
        boxes.sort(key=lambda item: item.timestamp_s)
    pose_times = [pose.tracking_timestamp_s for pose in camera_poses]
    output: dict[int, list[Observation]] = defaultdict(list)
    previous: dict[int, Observation] = {}
    previous_appearance: dict[int, np.ndarray] = {}
    for timestamp_s in times:
        pose_index = bisect.bisect_right(pose_times, timestamp_s) - 1
        if pose_index < 0:
            continue
        device_pose = camera_poses[pose_index]
        depth = np.asarray(depth_provider(timestamp_s))
        if depth.ndim != 2:
            raise ValueError("depth provider must return a two-dimensional array")
        camera_pose = Rigid(device_pose.t_device_m, device_pose.q_device_wxyz)
        camera_extrinsic = Rigid(calibration.t_device_m, calibration.q_device_wxyz)
        camera_translation = tuple(
            camera_pose.t_m[i]
            + quat_rotate(camera_pose.q_wxyz, camera_extrinsic.t_m)[i]
            for i in range(3)
        )
        camera_rotation = quat_mul(camera_pose.q_wxyz, camera_extrinsic.q_wxyz)
        camera_to_world = Rigid(camera_translation, camera_rotation)
        rgb = np.asarray(rgb_provider(timestamp_s)) if rgb_provider else None
        for object_uid, boxes in boxes_by_object.items():
            index = (
                bisect.bisect_right([box.timestamp_s for box in boxes], timestamp_s) - 1
            )
            if index < 0:
                continue
            box = boxes[index]
            if mask_provider is not None:
                mask = np.asarray(mask_provider(timestamp_s, object_uid), dtype=bool)
                if mask.shape != depth.shape:
                    raise ValueError("mask provider shape does not match depth")
            else:
                mask = np.zeros(depth.shape, dtype=bool)
                left = max(0, int(math.floor(box.x_min_px)))
                right = min(depth.shape[1], int(math.ceil(box.x_max_px)))
                top = max(0, int(math.floor(box.y_min_px)))
                bottom = min(depth.shape[0], int(math.ceil(box.y_max_px)))
                if left >= right or top >= bottom:
                    continue
                mask[top:bottom, left:right] = True
            if not mask.any():
                continue
            appearance = 1.0
            if rgb is not None:
                signature = masked_color_signature(rgb, mask)
                prior_signature = previous_appearance.get(object_uid)
                if prior_signature is not None:
                    appearance = cosine_similarity(signature, prior_signature)
                previous_appearance[object_uid] = signature
            try:
                observation = observation_from_rgbd(
                    sequence_id=seq.sequence_id,
                    frame_id=f"kf{len(output[object_uid]):06d}",
                    timestamp_s=round(timestamp_s, 6),
                    instance_id=str(object_uid),
                    depth_m=depth,
                    mask=mask,
                    camera_to_world=camera_to_world,
                    pinhole=calibration.pinhole,
                    co_visibility=1.0 if box.visibility_ratio >= 0.5 else 0.5,
                    appearance_similarity=appearance,
                    mask_area_px=box.area_px,
                )
            except ValueError:
                continue
            prior = previous.get(object_uid)
            if prior is not None:
                observation = replace(
                    observation,
                    displacement_m=round(
                        math.dist(observation.centroid_world_m, prior.centroid_world_m),
                        6,
                    ),
                )
            output[object_uid].append(observation)
            previous[object_uid] = observation
    return {object_uid: track for object_uid, track in output.items() if track}


def rgb_camera_calibration(seq: AdtSequence) -> CameraCalibration:
    """Return the RGB camera calibration record."""
    for label in ("camera-rgb",):
        if label in seq.calibrations:
            return seq.calibrations[label]
    for calibration in seq.calibrations.values():
        if "rgb" in calibration.label:
            return calibration
    raise ValueError(f"no RGB calibration in {seq.sequence_id}")
