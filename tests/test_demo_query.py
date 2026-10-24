"""Tests for the demo object query."""

from __future__ import annotations

import pytest

from egopersist.contracts import LabelClass
from egopersist.visualization import (
    ReplayMapEntry,
    answer_object_query,
    build_demo_replay,
)


def test_query_returns_state_position_and_confidence() -> None:
    scenario = build_demo_replay().scenario("durable_mug_move")
    frame = scenario.frames[-1]
    result = answer_object_query("mug", frame.ego_map, timestamp_s=frame.timestamp_s)
    assert result.object_id == "mug"
    assert result.state is LabelClass.STABLE
    assert result.position_m == (0.42, 0.05, 0.80)
    assert 0.0 <= result.confidence <= 1.0
    assert "0.42" in result.answer
    assert "confidence" in result.answer


def test_query_during_occlusion_reports_unknown() -> None:
    scenario = build_demo_replay().scenario("occlusion")
    frame = scenario.frames[6]
    result = answer_object_query("mug", frame.ego_map, timestamp_s=frame.timestamp_s)
    assert result.state is LabelClass.UNKNOWN
    assert "not fully visible" in result.answer


def test_query_matches_display_name_without_case() -> None:
    result = answer_object_query(
        "MUG",
        [
            ReplayMapEntry(
                "mug", "Mug", "mug", (1.0, 2.0, 3.0), LabelClass.STABLE, 0.8, 1
            )
        ],
        timestamp_s=1.0,
    )
    assert result.position_m == (1.0, 2.0, 3.0)


def test_query_rejects_missing_object() -> None:
    with pytest.raises(LookupError, match="not found"):
        answer_object_query("mug", [], timestamp_s=1.0)


def test_query_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        answer_object_query(
            "mug",
            [
                ReplayMapEntry(
                    "mug", "Mug", "mug", (1.0, 2.0, 3.0), LabelClass.STABLE, 0.8, 1
                )
            ],
            timestamp_s=-1.0,
        )
