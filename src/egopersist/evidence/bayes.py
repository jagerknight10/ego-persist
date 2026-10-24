"""Causal log-odds evidence accumulator for the four track labels.

Pure Python, no torch. Consumes the 20 causal features from
``features.py`` and produces per-keyframe class probabilities that the
state machine (``egopersist.mapping``) can drive with. Every update is
causal: only the current and past keyframes matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from egopersist.contracts import LabelClass
from egopersist.evidence.features import (
    FEATURE_NAMES,
    INPUT_DIM,
    _CHANGE_THRESHOLD_M,
)

_CLASS_ORDER: tuple[LabelClass, ...] = (
    LabelClass.STABLE,
    LabelClass.TRANSIENT,
    LabelClass.PERSISTENT_CHANGE,
    LabelClass.UNKNOWN,
)

_F_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def feature_scores(
    values: tuple[float, ...], dynamic: bool
) -> tuple[float, float, float, float]:
    """Per-class log-odds evidence for one keyframe.

    Order: (STABLE, TRANSIENT, PERSISTENT_CHANGE, UNKNOWN).
    Displacement/rotation evidence is the object's own motion and its
    deviation from the committed map pose; GT pose is never used.
    """
    if len(values) != INPUT_DIM:
        raise ValueError("expected one feature vector of length 20")
    disp = values[_F_IDX["displacement_m"]]
    rot = values[_F_IDX["rotation_deg"]] * 45.0  # denormalise
    dev = values[_F_IDX["deviation_from_committed_m"]]
    dev_rot = values[_F_IDX["deviation_from_committed_rot_deg"]] * 45.0
    speed = values[_F_IDX["speed_mps"]]
    co_vis = values[_F_IDX["co_visibility"]]
    depth = values[_F_IDX["depth_support"]]

    stable = 0.5 + 2.0 * (1.0 - min(1.0, dev / (2.0 * _CHANGE_THRESHOLD_M)))
    if co_vis < 0.6:
        stable -= 0.5
    if depth < 0.35:
        stable -= 0.5

    transient = 0.2
    if dynamic:
        transient += 1.5
    if speed > 0.3:
        transient += 1.0

    changed = 0.1
    if dev > _CHANGE_THRESHOLD_M or dev_rot > 15.0:
        changed += 1.2
    if max(disp, dev) > 2.0 * _CHANGE_THRESHOLD_M:
        changed += 1.0

    unknown = 0.3
    if co_vis < 0.6:
        unknown += 1.0
    if depth < 0.35:
        unknown += 0.5
    return (stable, transient, changed, unknown)


@dataclass
class BayesAccumulator:
    """Windowed log-odds accumulation for one track.

    ``window`` counts keyframes; the accumulator keeps the sum of
    per-keyframe scores over the most recent window and emits the
    softmax of that sum. The prior is uniform (zero log-odds).
    """

    window: int = 5
    _queue: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")

    def observe(self, values: tuple[float, ...], dynamic: bool) -> None:
        """Add one keyframe of features (FIFO window, exact)."""
        scores = feature_scores(values, dynamic)
        self._queue.append(scores)
        if len(self._queue) > self.window:
            self._queue.pop(0)

    def probabilities(self) -> dict[LabelClass, float]:
        """Softmax of the accumulated log-odds (finite, sums to 1)."""
        sums = [
            sum(scores[i] for scores in self._queue) for i in range(4)
        ] if self._queue else [0.0, 0.0, 0.0, 0.0]
        m = max(sums)
        exps = [math.exp(s - m) for s in sums]
        total = sum(exps)
        return {
            label: value / total for label, value in zip(_CLASS_ORDER, exps)
        }
