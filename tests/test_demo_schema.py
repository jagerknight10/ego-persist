"""Tests for the deterministic replay schema and scenario fixtures."""

from __future__ import annotations

import json

import pytest

from egopersist.contracts import DecisionAction, LabelClass
from egopersist.visualization import (
    SCHEMA_VERSION,
    ReplayDocument,
    ReplaySchemaError,
    build_demo_replay,
)


def test_replay_json_round_trip() -> None:
    replay = build_demo_replay()
    restored = ReplayDocument.from_json(replay.to_json())
    assert restored == replay


def test_replay_schema_version() -> None:
    replay = build_demo_replay()
    assert replay.schema_version == SCHEMA_VERSION
    assert json.loads(replay.to_json())["schema_version"] == SCHEMA_VERSION


def test_replay_is_deterministic() -> None:
    assert build_demo_replay().to_json() == build_demo_replay().to_json()


def test_replay_has_three_required_scenarios() -> None:
    replay = build_demo_replay()
    assert [item.scenario_id for item in replay.scenarios] == [
        "passing_person",
        "durable_mug_move",
        "occlusion",
    ]
    assert all(len(item.frames) >= 6 for item in replay.scenarios)


def test_replay_rejects_unknown_field() -> None:
    data = build_demo_replay().to_dict()
    data["future"] = True
    with pytest.raises(ReplaySchemaError, match="unknown fields"):
        ReplayDocument.from_dict(data)


def test_passing_person_does_not_commit_or_add_person() -> None:
    scenario = build_demo_replay().scenario("passing_person")
    for frame in scenario.frames:
        assert "person" not in {item.object_id for item in frame.ego_map}
        person_decisions = [
            item for item in frame.decisions if item.object_id == "person"
        ]
        if person_decisions:
            assert person_decisions[0].action is DecisionAction.REJECT
    assert "person" in {item.object_id for item in scenario.frames[-1].instant_map}
    assert scenario.frames[-1].query.state is LabelClass.STABLE
    assert scenario.frames[-1].query.position_m == (0.0, 0.0, 0.80)


def test_durable_move_defers_then_commits() -> None:
    scenario = build_demo_replay().scenario("durable_mug_move")
    transitions = [
        (frame, item)
        for frame in scenario.frames
        for item in frame.decisions
        if item.object_id == "mug"
        and item.action is DecisionAction.COMMIT
    ]
    assert len(transitions) == 1
    commit_frame, commit = transitions[0]
    assert commit.label is LabelClass.PERSISTENT_CHANGE
    assert commit_frame.timestamp_s == 7.5
    first_change = next(
        frame
        for frame in scenario.frames
        if frame.timestamp_s == 2.5
    )
    first_decision = next(
        item for item in first_change.decisions if item.object_id == "mug"
    )
    assert first_decision.action is DecisionAction.DEFER
    assert first_decision.instant_action is DecisionAction.COMMIT
    assert scenario.frames[-1].query.position_m == (0.42, 0.05, 0.80)
    assert scenario.frames[-1].query.state is LabelClass.STABLE


def test_occlusion_does_not_remove_mug() -> None:
    scenario = build_demo_replay().scenario("occlusion")
    hidden_frames = [frame for frame in scenario.frames if 2.0 <= frame.timestamp_s < 5.0]
    assert hidden_frames
    for frame in hidden_frames:
        assert "mug" in {item.object_id for item in frame.ego_map}
        assert "mug" not in {item.object_id for item in frame.instant_map}
        decision = next(item for item in frame.decisions if item.object_id == "mug")
        assert decision.action is DecisionAction.DEFER
        assert decision.label is LabelClass.UNKNOWN
        assert frame.query.position_m == (0.0, 0.0, 0.80)
    assert scenario.frames[-1].query.state is LabelClass.STABLE
