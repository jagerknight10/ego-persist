"""Pareto plot data for map-maintenance systems."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One system result in the false-commit/stale-map plane."""

    system_id: str
    transient_false_commit_rate: float
    stale_map_duration_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.system_id, str) or not self.system_id:
            raise ValueError("system_id must be a non-empty string")
        for name in ("transient_false_commit_rate", "stale_map_duration_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, number)

    def to_dict(self) -> dict[str, str | float]:
        """Return a JSON-compatible dictionary."""
        return {
            "system_id": self.system_id,
            "transient_false_commit_rate": self.transient_false_commit_rate,
            "stale_map_duration_s": self.stale_map_duration_s,
        }


def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
    return (
        left.transient_false_commit_rate <= right.transient_false_commit_rate
        and left.stale_map_duration_s <= right.stale_map_duration_s
        and (
            left.transient_false_commit_rate < right.transient_false_commit_rate
            or left.stale_map_duration_s < right.stale_map_duration_s
        )
    )


def pareto_plot_data(points: Iterable[ParetoPoint]) -> dict[str, object]:
    """Return sorted plot data and the non-dominated frontier.

    Both axes are minimized: transient false-commit rate and stale-map
    duration in seconds.
    """
    point_list = tuple(
        sorted(
            points,
            key=lambda point: (
                point.transient_false_commit_rate,
                point.stale_map_duration_s,
                point.system_id,
            ),
        )
    )
    frontier = tuple(
        point
        for point in point_list
        if not any(
            _dominates(other, point) for other in point_list if other is not point
        )
    )
    return {
        "x_metric": "transient_false_commit_rate",
        "y_metric": "stale_map_duration_s",
        "points": [point.to_dict() for point in point_list],
        "frontier": [point.to_dict() for point in frontier],
    }
