"""Deterministic keyframe sampling and Observation construction.

Keyframes are the sequence start plus integer multiples of ``1 / hz``.
Sampling never looks ahead: the pose used at keyframe time t is the last
recorded pose with timestamp <= t.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict

from wise_egomap.contracts import Observation
from wise_egomap.data.adt import AdtSequence, CameraCalibration

RGB_STREAM_CANDIDATES = ("1201-1", "1201-2")


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


def rgb_camera_calibration(seq: AdtSequence) -> CameraCalibration:
    """Return the RGB camera calibration record."""
    for label in ("camera-rgb",):
        if label in seq.calibrations:
            return seq.calibrations[label]
    for calibration in seq.calibrations.values():
        if "rgb" in calibration.label:
            return calibration
    raise ValueError(f"no RGB calibration in {seq.sequence_id}")
