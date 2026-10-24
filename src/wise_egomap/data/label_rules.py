"""Rule engine that turns track pose timelines into event labels.

Rules come from ``configs/labels_v1.yaml``. The engine is deterministic and
causal: the label at keyframe k depends only on keyframes with index <= k.
Ground-truth pose is a label input, never a model feature.

Per-keyframe classes:
- ``STABLE``: static object at its reference position.
- ``TRANSIENT``: dynamic object (person, held item) in the scene.
- ``PERSISTENT_CHANGE``: static object confirmed at a new position.
- ``UNKNOWN``: not visible, or visible but the move is not yet confirmed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from wise_egomap.contracts import EventLabel, EventType, LabelClass


@dataclass(frozen=True)
class LabelConfig:
    """Thresholds for the label-rule engine. Metres, degrees, seconds."""

    keyframe_hz: float
    change_translation_m: float
    change_rotation_deg: float
    persistent_hold_s: float
    persistent_min_visible_keyframes: int
    persistent_mode_tolerance_m: float
    persistent_mode_tolerance_deg: float
    transient_return_window_s: float
    transient_return_tolerance_m: float
    transient_return_tolerance_deg: float
    visibility_min_mask_pixels: int
    visibility_min_depth_support: float
    visibility_co_visibility_threshold: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> LabelConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if raw.get("version") != 1:
            raise ValueError("unsupported labels config version")
        return cls(
            keyframe_hz=float(raw["keyframe_hz"]),
            change_translation_m=float(raw["change"]["translation_m"]),
            change_rotation_deg=float(raw["change"]["rotation_deg"]),
            persistent_hold_s=float(raw["persistent"]["hold_s"]),
            persistent_min_visible_keyframes=int(
                raw["persistent"]["min_visible_keyframes"]
            ),
            persistent_mode_tolerance_m=float(raw["persistent"]["mode_tolerance_m"]),
            persistent_mode_tolerance_deg=float(
                raw["persistent"]["mode_tolerance_deg"]
            ),
            transient_return_window_s=float(raw["transient"]["return_window_s"]),
            transient_return_tolerance_m=float(raw["transient"]["return_tolerance_m"]),
            transient_return_tolerance_deg=float(
                raw["transient"]["return_tolerance_deg"]
            ),
            visibility_min_mask_pixels=int(raw["visibility"]["min_mask_pixels"]),
            visibility_min_depth_support=float(raw["visibility"]["min_depth_support"]),
            visibility_co_visibility_threshold=float(
                raw["visibility"]["co_visibility_threshold"]
            ),
        )


def load_label_config(path: str | Path) -> LabelConfig:
    return LabelConfig.from_yaml(path)


@dataclass(frozen=True)
class TrackSample:
    """One keyframe of one track.

    ``pose`` and ``orientation_wxyz`` are None when the object is not
    visible at this keyframe.
    """

    timestamp_s: float
    pose: tuple[float, float, float] | None
    orientation_wxyz: tuple[float, float, float, float] | None
    visible: bool
    dynamic: bool


def _angle_between_wxyz(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    dot = max(-1.0, min(1.0, abs(sum(x * y for x, y in zip(a, b, strict=False)))))
    return math.degrees(2.0 * math.acos(dot))


def sample_is_transient_track(samples: list[TrackSample]) -> bool:
    """A track is transient when every one of its samples is dynamic.

    ADT marks moving people as ``motion_type = dynamic``. A static object
    that is moved produces a ``PERSISTENT_CHANGE`` window instead.
    """
    if not samples:
        return False
    return all(sample.dynamic for sample in samples)


def label_track(
    sequence_id: str,
    track_id: str,
    samples: list[TrackSample],
    config: LabelConfig,
) -> tuple[list[EventLabel], list[LabelClass]]:
    """Label one track timeline.

    Returns the event windows and the per-keyframe class list. Both are
    causal: the output at keyframe k uses only samples with index <= k.
    """
    if not samples:
        raise ValueError("track has no keyframes")
    if any(
        samples[i].timestamp_s >= samples[i + 1].timestamp_s
        for i in range(len(samples) - 1)
    ):
        raise ValueError("samples must be strictly increasing in time")

    events: list[EventLabel] = []
    per_keyframe: list[LabelClass] = [LabelClass.UNKNOWN] * len(samples)

    if sample_is_transient_track(samples):
        visible = [i for i, s in enumerate(samples) if s.visible]
        for index in visible:
            per_keyframe[index] = LabelClass.TRANSIENT
        if visible:
            events.append(
                EventLabel(
                    sequence_id=sequence_id,
                    track_id=track_id,
                    event_type=EventType.TRANSIENT,
                    start_s=samples[visible[0]].timestamp_s,
                    end_s=samples[visible[-1]].timestamp_s,
                    label=LabelClass.TRANSIENT,
                )
            )
        return events, per_keyframe

    anchor: tuple[float, float, float] | None = None
    anchor_ori: tuple[float, float, float, float] | None = None
    state = "stable"
    change_start: int | None = None
    change_anchor: tuple[float, float, float] | None = None

    for index, sample in enumerate(samples):
        if not sample.visible or sample.pose is None:
            per_keyframe[index] = LabelClass.UNKNOWN
            continue
        pose = sample.pose
        orientation = sample.orientation_wxyz
        if orientation is None:
            per_keyframe[index] = LabelClass.UNKNOWN
            continue

        if anchor is None:
            anchor = pose
            anchor_ori = orientation
            per_keyframe[index] = LabelClass.STABLE
            continue

        distance = math.dist(pose, anchor)
        rotation = _angle_between_wxyz(orientation, anchor_ori)

        if state == "stable":
            per_keyframe[index] = LabelClass.STABLE
            if (
                distance > config.change_translation_m
                or rotation > config.change_rotation_deg
            ):
                state = "suspect"
                change_start = index
                change_anchor = pose
        elif state == "suspect":
            per_keyframe[index] = LabelClass.UNKNOWN
            assert change_anchor is not None and change_start is not None
            returned = (
                distance <= config.transient_return_tolerance_m
                and rotation <= config.transient_return_tolerance_deg
            )
            if returned:
                events.append(
                    EventLabel(
                        sequence_id=sequence_id,
                        track_id=track_id,
                        event_type=EventType.REVERSION,
                        start_s=samples[change_start].timestamp_s,
                        end_s=sample.timestamp_s,
                        label=LabelClass.STABLE,
                    )
                )
                state = "stable"
                continue
            hold_start_t = samples[change_start].timestamp_s
            if sample.timestamp_s < hold_start_t + config.persistent_hold_s:
                continue
            window = [
                i
                for i in range(change_start, index + 1)
                if samples[i].visible
                and samples[i].pose is not None
                and samples[i].orientation_wxyz is not None
            ]
            if len(window) < config.persistent_min_visible_keyframes:
                continue
            poses = [samples[i].pose for i in window]
            oris = [samples[i].orientation_wxyz for i in window]
            mode_pose = tuple(sum(p) / len(p) for p in zip(*poses, strict=False))
            stable_pose = all(
                math.dist(p, mode_pose) <= config.persistent_mode_tolerance_m
                for p in poses
            )
            stable_ori = all(
                _angle_between_wxyz(o, oris[-1]) <= config.persistent_mode_tolerance_deg
                for o in oris
            )
            if stable_pose and stable_ori:
                events.append(
                    EventLabel(
                        sequence_id=sequence_id,
                        track_id=track_id,
                        event_type=EventType.PERSISTENT_CHANGE,
                        start_s=samples[change_start].timestamp_s,
                        end_s=sample.timestamp_s,
                        label=LabelClass.PERSISTENT_CHANGE,
                    )
                )
                state = "changed"
                anchor = pose
                anchor_ori = orientation
                per_keyframe[index] = LabelClass.PERSISTENT_CHANGE
            else:
                state = "stable"
                per_keyframe[index] = LabelClass.STABLE
        elif state == "changed":
            per_keyframe[index] = LabelClass.PERSISTENT_CHANGE
            if (
                distance > config.change_translation_m
                or rotation > config.change_rotation_deg
            ):
                state = "suspect"
                change_start = index
                change_anchor = pose

    stable_keys = [
        i for i in range(len(samples)) if per_keyframe[i] is LabelClass.STABLE
    ]
    if stable_keys:
        events.append(
            EventLabel(
                sequence_id=sequence_id,
                track_id=track_id,
                event_type=EventType.STABLE,
                start_s=samples[stable_keys[0]].timestamp_s,
                end_s=samples[stable_keys[-1]].timestamp_s,
                label=LabelClass.STABLE,
            )
        )
    return events, per_keyframe
