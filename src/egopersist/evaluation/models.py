"""Typed values used by the event evaluator.

All distances use metres. All timestamps and durations use seconds. The
evaluator uses ground-truth event windows and map actions. Ground truth is
never a model input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from egopersist.contracts import Decision, DecisionAction, MapUpdate

Position3 = tuple[float, float, float]


def _finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _time(value: float, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _position(value: Position3 | None, name: str) -> Position3 | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must have three values")
    x = _finite(value[0], f"{name}[0]")
    y = _finite(value[1], f"{name}[1]")
    z = _finite(value[2], f"{name}[2]")
    return (x, y, z)


@dataclass(frozen=True, slots=True)
class ActionEvent:
    """One causal map action.

    A decision supplies an action without a map position. A map update
    supplies an action and, for commits and reverts, a new position.
    """

    sequence_id: str
    track_id: str
    timestamp_s: float
    action: DecisionAction
    frame_id: str = ""
    previous_position_m: Position3 | None = None
    new_position_m: Position3 | None = None

    def __post_init__(self) -> None:
        for name in ("sequence_id", "track_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.frame_id, str):
            raise TypeError("frame_id must be a string")
        object.__setattr__(self, "timestamp_s", _time(self.timestamp_s, "timestamp_s"))
        if not isinstance(self.action, DecisionAction):
            raise TypeError("action must be a DecisionAction")
        object.__setattr__(
            self,
            "previous_position_m",
            _position(self.previous_position_m, "previous_position_m"),
        )
        object.__setattr__(
            self, "new_position_m", _position(self.new_position_m, "new_position_m")
        )

    @classmethod
    def from_decision(cls, decision: Decision) -> ActionEvent:
        """Make an action event from a decision."""
        return cls(
            sequence_id=decision.sequence_id,
            track_id=decision.track_id,
            timestamp_s=decision.timestamp_s,
            action=decision.action,
            frame_id=decision.frame_id,
        )

    @classmethod
    def from_update(cls, update: MapUpdate) -> ActionEvent:
        """Make an action event from a map update."""
        return cls(
            sequence_id=update.sequence_id,
            track_id=update.track_id,
            timestamp_s=update.timestamp_s,
            action=update.action,
            previous_position_m=update.previous_position_m,
            new_position_m=update.new_position_m,
        )

    @classmethod
    def from_record(cls, record: Decision | MapUpdate) -> ActionEvent:
        """Make an action event from a decision or map update."""
        if isinstance(record, Decision):
            return cls.from_decision(record)
        if isinstance(record, MapUpdate):
            return cls.from_update(record)
        raise TypeError("record must be a Decision or MapUpdate")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "track_id": self.track_id,
            "timestamp_s": self.timestamp_s,
            "action": self.action.value,
            "frame_id": self.frame_id,
            "previous_position_m": self.previous_position_m,
            "new_position_m": self.new_position_m,
        }


@dataclass(frozen=True, slots=True)
class PositionReference:
    """Ground-truth object position at one timestamp."""

    sequence_id: str
    track_id: str
    timestamp_s: float
    position_m: Position3

    def __post_init__(self) -> None:
        for name in ("sequence_id", "track_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "timestamp_s", _time(self.timestamp_s, "timestamp_s"))
        position = _position(self.position_m, "position_m")
        if position is None:
            raise ValueError("position_m must not be None")
        object.__setattr__(self, "position_m", position)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "track_id": self.track_id,
            "timestamp_s": self.timestamp_s,
            "position_m": self.position_m,
        }


def distance_m(left: Position3, right: Position3) -> float:
    """Return the Euclidean distance between two positions in metres."""
    return math.dist(left, right)


@dataclass(frozen=True, slots=True)
class RateMetric:
    """An event-level rate."""

    numerator: int
    denominator: int
    rate: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible dictionary."""
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.rate,
        }


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Counts and scores for persistent-change detection."""

    true_positives: int
    false_positives: int
    false_negatives: int
    predicted_positives: int
    actual_positives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible dictionary."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "predicted_positives": self.predicted_positives,
            "actual_positives": self.actual_positives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


class ErrorCategory(StrEnum):
    """Stable error categories for false map actions."""

    TRANSIENT_FALSE_COMMIT = "transient_false_commit"
    UNKNOWN_EVENT_COMMIT = "unknown_event_commit"
    UNKNOWN_FALSE_REMOVAL = "unknown_false_removal"
    DUPLICATE_COMMIT = "duplicate_commit"
    UNMATCHED_COMMIT = "unmatched_commit"
    MISSED_PERSISTENT_CHANGE = "missed_persistent_change"


EMPTY_CASE_POLICY: dict[str, str | float | None] = {
    "no_predicted_positives_precision": 0.0,
    "no_actual_positives_recall": 1.0,
    "undefined_f1": 0.0,
    "zero_denominator_rate": 0.0,
    "no_timing_values_mean_s": None,
    "no_stale_events_total_duration_s": 0.0,
    "no_stale_events_mean_stale_map_duration_s": 0.0,
    "no_position_errors_mean_m": None,
    "no_update_utility_values_mean": 0.0,
    "no_taxonomy_items_count": 0,
    "no_pareto_points_count": 0,
}


def empty_case_policy() -> dict[str, str | float | None]:
    """Return a copy of the documented empty-case policy."""
    return dict(EMPTY_CASE_POLICY)
