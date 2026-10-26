"""Small tests for the real-ADT video overlay helpers."""

from wise_egomap.data.adt import Box2D
from wise_egomap.visualization.adt_video import (
    DecisionTrace,
    action_color,
    boxes_near_time,
    nearest_decision,
)


def test_nearest_decision_respects_time_tolerance() -> None:
    trace = (DecisionTrace(1.0, "keep", "STABLE", 0.9, {}),)
    assert nearest_decision(trace, 1.4) is not None
    assert nearest_decision(trace, 1.7) is None


def test_boxes_near_time_returns_largest_first() -> None:
    small = Box2D("214-1", 1, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0)
    large = Box2D("214-1", 2, 1.1, 0.0, 4.0, 0.0, 4.0, 1.0)
    assert [box.object_uid for box in boxes_near_time((small, large), 1.0)] == [2, 1]


def test_action_colours_are_stable() -> None:
    assert action_color("commit") == "#ff7a59"
    assert action_color("unknown") == "#e8eef0"
