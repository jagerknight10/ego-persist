"""Causal feature extraction for one track observation.

Every value uses only the current observation and the track's own past.
Ground-truth pose is never a feature: the only position-related inputs are
the object's *committed* map pose and the object's own prior observations.
The feature vector has exactly ``INPUT_DIM`` values, in ``FEATURE_NAMES``
order. All scalars are finite.
"""

from __future__ import annotations

import math

from wise_egomap.contracts import Observation, TrackObservation

INPUT_DIM = 20

FEATURE_NAMES: tuple[str, ...] = (
    "visible",
    "motion_type_dynamic",
    "dt_since_last_s",
    "displacement_m",
    "rotation_deg",
    "speed_mps",
    "angle_change_from_mean_deg",
    "deviation_from_committed_m",
    "deviation_from_committed_rot_deg",
    "ratio_to_change_threshold",
    "mask_area_px_norm",
    "mask_area_delta_norm",
    "depth_support",
    "co_visibility",
    "appearance_similarity",
    "consecutive_visible",
    "visible_fraction_16",
    "mean_speed_8_mps",
    "mean_deviation_8_m",
    "time_since_last_change_s",
)

assert len(FEATURE_NAMES) == INPUT_DIM
assert len(set(FEATURE_NAMES)) == INPUT_DIM

_MASK_AREA_PX_NORM = 40_000.0  # 4 m x 10 m mask; normalisation only
_CHANGE_THRESHOLD_M = 0.10  # mirrors configs/labels_v1.yaml
_ANGLE_NORM_DEG = 45.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _unit(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("feature input must be finite")
    return value


def _angle_between_wxyz(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    dot = max(-1.0, min(1.0, abs(sum(x * y for x, y in zip(a, b, strict=False)))))
    return math.degrees(2.0 * math.acos(dot))


class TrackFeatureState:
    """Causal memory of one track. Only past and present data."""

    def __init__(self) -> None:
        self.last_timestamp_s: float | None = None
        self.last_obs: Observation | None = None
        self.last_committed_dev_m: float | None = None
        self.consecutive_visible = 0
        self.visible_window: list[bool] = []  # up to 16 keyframes
        self.speed_window: list[float] = []  # up to 8
        self.dev_window: list[float] = []  # up to 8
        self.mean_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self.mean_orientation_weight = 0.0
        self.last_change_timestamp_s: float = 0.0
        self.has_change_ever = False

    def _update_mean_orientation(
        self, orientation: tuple[float, float, float, float]
    ) -> None:
        """Approximate running mean via weighted vector sum + renormalise."""
        weight = self.mean_orientation_weight + 1.0
        old = self.mean_orientation
        total = [
            old[i] * self.mean_orientation_weight + orientation[i] for i in range(4)
        ]
        norm = math.sqrt(sum(v * v for v in total))
        if norm > 1e-12:
            self.mean_orientation = tuple(v / norm for v in total)  # type: ignore[assignment]
        self.mean_orientation_weight = weight

    def update(self, view: TrackObservation, dynamic: bool) -> tuple[float, ...]:
        """Compute features for this view, then advance the causal memory."""
        obs = view.observation
        dt = (
            obs.timestamp_s - self.last_timestamp_s
            if self.last_timestamp_s is not None
            else 0.0
        )
        speed = obs.displacement_m / dt if dt > 1e-9 else 0.0
        committed_dev_m = math.dist(obs.centroid_world_m, view.committed_position_m)
        committed_dev_rot = _angle_between_wxyz(
            obs.orientation_wxyz, view.committed_orientation_wxyz
        )
        ratio = _clamp(max(obs.displacement_m, committed_dev_m) / _CHANGE_THRESHOLD_M)

        mask_norm = _clamp(obs.mask_area_px / _MASK_AREA_PX_NORM)
        if self.last_obs is not None:
            delta_px = obs.mask_area_px - self.last_obs.mask_area_px
            mask_delta = _clamp(abs(delta_px) / _MASK_AREA_PX_NORM)
        else:
            mask_delta = 0.0

        if self.mean_orientation_weight > 0.0:
            angle_from_mean = _clamp(
                _angle_between_wxyz(obs.orientation_wxyz, self.mean_orientation)
                / _ANGLE_NORM_DEG
            )
        else:
            angle_from_mean = 0.0

        visible_fraction = (
            sum(self.visible_window) / len(self.visible_window)
            if self.visible_window
            else 1.0
        )
        mean_speed = (
            sum(self.speed_window) / len(self.speed_window)
            if self.speed_window
            else 0.0
        )
        mean_dev = (
            sum(self.dev_window) / len(self.dev_window) if self.dev_window else 0.0
        )
        time_since_change = (
            obs.timestamp_s - self.last_change_timestamp_s
            if self.has_change_ever
            else 0.0
        )

        values = (
            _unit(1.0),
            _unit(1.0) if dynamic else _unit(0.0),
            _clamp(dt / 10.0),
            _clamp(obs.displacement_m),
            _clamp(obs.rotation_deg / _ANGLE_NORM_DEG),
            _clamp(speed),
            angle_from_mean,
            _clamp(committed_dev_m),
            _clamp(committed_dev_rot / _ANGLE_NORM_DEG),
            ratio,
            mask_norm,
            mask_delta,
            _clamp(obs.depth_support),
            _clamp(obs.co_visibility),
            _clamp(obs.appearance_similarity),
            _clamp(self.consecutive_visible / 16.0),
            visible_fraction,
            _clamp(mean_speed),
            _clamp(mean_dev),
            _clamp(time_since_change / 30.0),
        )
        assert len(values) == INPUT_DIM

        # advance causal memory
        if not self.has_change_ever and obs.displacement_m > _CHANGE_THRESHOLD_M:
            self.has_change_ever = True
            self.last_change_timestamp_s = obs.timestamp_s

        self.last_timestamp_s = obs.timestamp_s
        self.last_obs = obs
        self.last_committed_dev_m = committed_dev_m
        self.consecutive_visible = self.consecutive_visible + 1
        self.visible_window = (self.visible_window + [True])[-16:]
        self.speed_window = (self.speed_window + [speed])[-8:]
        self.dev_window = (self.dev_window + [committed_dev_m])[-8:]
        self._update_mean_orientation(obs.orientation_wxyz)
        return values


def features_for_observation(
    view: TrackObservation,
    dynamic: bool,
    state: TrackFeatureState,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    values = state.update(view, dynamic)
    return FEATURE_NAMES, values
