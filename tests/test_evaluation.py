"""CPU tests for deterministic EgoPersist evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from egopersist.contracts import (
    Decision,
    DecisionAction,
    EventLabel,
    EventType,
    LabelClass,
    MapUpdate,
)
from egopersist.evaluation import (
    ActionEvent,
    EMPTY_CASE_POLICY,
    ErrorCategory,
    ParetoPoint,
    PositionReference,
    classify_errors,
    commit_timing,
    evaluation_report,
    final_position_error_m,
    match_events,
    pareto_plot_data,
    persistent_change_metrics,
    transient_false_commit_rate,
    unknown_false_removal_rate,
    update_utility_at_0_30_m,
    write_report,
)


def event(
    event_type: EventType,
    start_s: float,
    end_s: float,
    *,
    sequence_id: str = "seq",
    track_id: str = "mug",
) -> EventLabel:
    label = {
        EventType.STABLE: LabelClass.STABLE,
        EventType.TRANSIENT: LabelClass.TRANSIENT,
        EventType.PERSISTENT_CHANGE: LabelClass.PERSISTENT_CHANGE,
        EventType.OCCLUSION: LabelClass.UNKNOWN,
        EventType.REVERSION: LabelClass.STABLE,
    }[event_type]
    return EventLabel(
        sequence_id=sequence_id,
        track_id=track_id,
        event_type=event_type,
        start_s=start_s,
        end_s=end_s,
        label=label,
    )


def move(
    timestamp_s: float,
    position_m: tuple[float, float, float],
    *,
    previous_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sequence_id: str = "seq",
    track_id: str = "mug",
) -> MapUpdate:
    return MapUpdate(
        sequence_id=sequence_id,
        track_id=track_id,
        timestamp_s=timestamp_s,
        action=DecisionAction.COMMIT,
        map_version=int(timestamp_s * 10),
        previous_position_m=previous_m,
        new_position_m=position_m,
    )


def reference(
    timestamp_s: float,
    position_m: tuple[float, float, float],
    *,
    sequence_id: str = "seq",
    track_id: str = "mug",
) -> PositionReference:
    return PositionReference(
        sequence_id=sequence_id,
        track_id=track_id,
        timestamp_s=timestamp_s,
        position_m=position_m,
    )


def protocol_events() -> tuple[EventLabel, ...]:
    return (
        event(EventType.TRANSIENT, 0.0, 5.0, track_id="person"),
        event(EventType.PERSISTENT_CHANGE, 10.0, 20.0),
        event(EventType.OCCLUSION, 25.0, 30.0, track_id="cup"),
    )


def test_known_small_example_metrics() -> None:
    events = protocol_events()
    actions = (
        ActionEvent("seq", "person", 1.0, DecisionAction.COMMIT),
        ActionEvent("seq", "mug", 12.0, DecisionAction.COMMIT),
        ActionEvent("seq", "cup", 26.0, DecisionAction.REJECT),
    )

    persistent = persistent_change_metrics(events, actions)
    transient = transient_false_commit_rate(events, actions)
    unknown = unknown_false_removal_rate(events, actions)

    assert persistent.true_positives == 1
    assert persistent.false_positives == 1
    assert persistent.false_negatives == 0
    assert persistent.precision == pytest.approx(0.5)
    assert persistent.recall == pytest.approx(1.0)
    assert persistent.f1 == pytest.approx(2.0 / 3.0)
    assert transient.to_dict() == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert unknown.to_dict() == {"numerator": 1, "denominator": 1, "rate": 1.0}


def test_perfect_prediction() -> None:
    events = protocol_events()
    actions = (
        ActionEvent("seq", "person", 2.0, DecisionAction.REJECT),
        move(12.0, (1.0, 0.0, 0.0)),
        ActionEvent("seq", "cup", 26.0, DecisionAction.DEFER),
    )

    persistent = persistent_change_metrics(events, actions)
    timing = commit_timing(events, actions)
    utility = update_utility_at_0_30_m(
        (move(12.0, (1.0, 0.0, 0.0)),),
        (
            reference(11.0, (0.0, 0.0, 0.0)),
            reference(12.0, (1.0, 0.0, 0.0)),
            reference(20.0, (1.0, 0.0, 0.0)),
        ),
    )

    assert persistent.to_dict() == {
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "predicted_positives": 1,
        "actual_positives": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert transient_false_commit_rate(events, actions).rate == 0.0
    assert unknown_false_removal_rate(events, actions).rate == 0.0
    assert timing.time_to_correct_commit_s == pytest.approx(2.0)
    assert timing.total_stale_map_duration_s == pytest.approx(2.0)
    assert timing.entries[0].stale_map_duration_s == pytest.approx(2.0)
    assert utility.mean_utility == 1.0
    assert utility.useful_count == 1


def test_never_update_case() -> None:
    events = protocol_events()
    actions = (
        ActionEvent("seq", "person", 2.0, DecisionAction.KEEP),
        ActionEvent("seq", "mug", 12.0, DecisionAction.KEEP),
        ActionEvent("seq", "cup", 26.0, DecisionAction.KEEP),
    )

    persistent = persistent_change_metrics(events, actions)
    timing = commit_timing(events, actions)
    taxonomy = classify_errors(events, actions)

    assert persistent.precision == 0.0
    assert persistent.recall == 0.0
    assert persistent.f1 == 0.0
    assert timing.time_to_correct_commit_s is None
    assert timing.total_stale_map_duration_s == pytest.approx(10.0)
    assert timing.entries[0].time_to_correct_commit_s is None
    assert taxonomy.counts == {"missed_persistent_change": 1}


def test_instant_update_case_vs_perfect() -> None:
    events = protocol_events()
    instant_actions = (
        ActionEvent("seq", "person", 0.0, DecisionAction.COMMIT),
        ActionEvent("seq", "mug", 10.0, DecisionAction.COMMIT),
        ActionEvent("seq", "cup", 25.0, DecisionAction.REJECT),
    )
    perfect_actions = (
        ActionEvent("seq", "person", 2.0, DecisionAction.REJECT),
        ActionEvent("seq", "mug", 12.0, DecisionAction.COMMIT),
        ActionEvent("seq", "cup", 26.0, DecisionAction.DEFER),
    )

    instant = evaluation_report(events, instant_actions, system_id="instant")
    perfect = evaluation_report(events, perfect_actions, system_id="perfect")

    assert instant["transient_false_commit"]["rate"] == 1.0
    assert perfect["transient_false_commit"]["rate"] == 0.0
    assert instant["commit_timing"]["time_to_correct_commit_s"] == 0.0
    assert perfect["commit_timing"]["time_to_correct_commit_s"] == 2.0
    assert instant["persistent_change_micro"]["precision"] == 0.5
    assert instant["persistent_change_micro"]["f1"] == pytest.approx(2.0 / 3.0)
    assert instant["error_taxonomy"]["counts"] == {
        "transient_false_commit": 1,
        "unknown_false_removal": 1
    }


def test_missing_event_case() -> None:
    actions = (ActionEvent("seq", "ghost", 5.0, DecisionAction.COMMIT),)

    persistent = persistent_change_metrics((), actions)
    taxonomy = classify_errors((), actions)

    assert persistent.actual_positives == 0
    assert persistent.predicted_positives == 1
    assert persistent.precision == 0.0
    assert persistent.recall == 1.0
    assert persistent.f1 == 0.0
    assert taxonomy.counts == {"unmatched_commit": 1}


def test_empty_cases_are_documented() -> None:
    persistent = persistent_change_metrics((), ())
    timing = commit_timing((), ())
    position = final_position_error_m((), ())
    utility = update_utility_at_0_30_m((), ())
    taxonomy = classify_errors((), ())

    assert persistent.f1 == EMPTY_CASE_POLICY["undefined_f1"]
    assert timing.time_to_correct_commit_s is None
    assert timing.total_stale_map_duration_s == 0.0
    assert position.mean_error_m is None
    assert utility.mean_utility == EMPTY_CASE_POLICY["no_update_utility_values_mean"]
    assert taxonomy.counts == {}
    assert taxonomy.entries == ()


def test_matching_is_one_to_one_and_deterministic() -> None:
    overlapping = EventLabel(
        sequence_id="seq",
        track_id="mug",
        event_type=EventType.PERSISTENT_CHANGE,
        start_s=0.0,
        end_s=10.0,
        label=LabelClass.PERSISTENT_CHANGE,
    )
    action_a = ActionEvent("seq", "mug", 2.0, DecisionAction.COMMIT)
    action_b = ActionEvent("seq", "mug", 4.0, DecisionAction.COMMIT)
    first = match_events((overlapping,), (action_b, action_a))
    second = match_events((overlapping,), (action_a, action_b))

    assert first == second
    assert len(first.matches) == 1
    assert len(first.unmatched_actions) == 1
    assert first.matches[0].action.timestamp_s == 2.0


def test_wrong_action_inside_window_does_not_match() -> None:
    change = event(EventType.PERSISTENT_CHANGE, 10.0, 20.0)
    result = match_events(
        (change,),
        (ActionEvent("seq", "mug", 12.0, DecisionAction.KEEP),),
    )

    assert result.matches == ()
    assert result.unmatched_events == (change,)
    assert len(result.unmatched_actions) == 1


def test_taxonomy_classifies_false_actions() -> None:
    events = protocol_events()
    actions = (
        ActionEvent("seq", "person", 1.0, DecisionAction.COMMIT),
        ActionEvent("seq", "mug", 12.0, DecisionAction.COMMIT),
        ActionEvent("seq", "mug", 14.0, DecisionAction.COMMIT),
        ActionEvent("seq", "cup", 26.0, DecisionAction.REJECT),
        ActionEvent("seq", "unknown", 40.0, DecisionAction.COMMIT),
    )

    taxonomy = classify_errors(events, actions)

    assert taxonomy.counts == {
        ErrorCategory.TRANSIENT_FALSE_COMMIT.value: 1,
        ErrorCategory.DUPLICATE_COMMIT.value: 1,
        ErrorCategory.UNKNOWN_FALSE_REMOVAL.value: 1,
        ErrorCategory.UNMATCHED_COMMIT.value: 1,
    }


def test_update_utility_penalizes_harmful_commit_and_uses_boundary() -> None:
    harmless = move(5.0, (0.30, 0.0, 0.0))
    harmful = move(15.0, (1.0, 0.0, 0.0), previous_m=(0.0, 0.0, 0.0))
    report = update_utility_at_0_30_m(
        (harmless, harmful),
        (reference(5.0, (0.0, 0.0, 0.0)), reference(15.0, (0.0, 0.0, 0.0))),
    )

    assert report.entries[0].utility == 0.0
    assert report.entries[1].utility == -1.0
    assert report.harmful_count == 1
    assert report.mean_utility == -0.5


def test_final_position_error_uses_causal_update() -> None:
    updates = (
        move(5.0, (0.1, 0.0, 0.0)),
        move(15.0, (2.0, 0.0, 0.0), previous_m=(0.1, 0.0, 0.0)),
    )
    report = final_position_error_m(
        updates,
        (reference(10.0, (0.0, 0.0, 0.0)),),
    )

    assert report.entries[0].error_m == pytest.approx(0.1)


def test_final_position_error_uses_latest_reference() -> None:
    report = final_position_error_m(
        (move(5.0, (0.1, 0.0, 0.0)),),
        (
            reference(20.0, (2.0, 0.0, 0.0)),
            reference(10.0, (0.0, 0.0, 0.0)),
        ),
    )

    assert report.entries[0].reference_timestamp_s == 20.0
    assert report.mean_error_m == pytest.approx(1.9)


def test_pareto_data_is_sorted_and_marks_frontier() -> None:
    data = pareto_plot_data(
        (
            ParetoPoint("never", 0.0, 10.0),
            ParetoPoint("instant", 0.5, 0.0),
            ParetoPoint("dominated", 0.6, 11.0),
            ParetoPoint("balanced", 0.2, 2.0),
        )
    )

    assert [point["system_id"] for point in data["points"]] == [
        "never",
        "balanced",
        "instant",
        "dominated",
    ]
    assert [point["system_id"] for point in data["frontier"]] == [
        "never",
        "balanced",
        "instant",
    ]


def test_report_is_machine_readable_and_stable(tmp_path: Path) -> None:
    events = protocol_events()
    actions = (
        ActionEvent("seq", "person", 1.0, DecisionAction.COMMIT),
        ActionEvent("seq", "mug", 12.0, DecisionAction.COMMIT),
        ActionEvent("seq", "cup", 26.0, DecisionAction.REJECT),
    )
    updates = (move(12.0, (1.0, 0.0, 0.0)),)
    refs = (
        reference(11.0, (0.0, 0.0, 0.0)),
        reference(12.0, (1.0, 0.0, 0.0)),
        reference(20.0, (1.0, 0.0, 0.0)),
    )
    report = evaluation_report(
        events,
        actions,
        updates=iter(updates),
        references=iter(refs),
        system_id="test",
    )
    path = write_report(report, tmp_path / "report.json")
    restored = json.loads(path.read_text(encoding="utf-8"))

    assert restored == report
    assert restored["schema_version"] == "evaluation_v1"
    assert restored["aggregation"]["mean_of_ratios_computed"] is False
    assert restored["update_utility_at_0_30_m"]["threshold_m"] == 0.30
    assert restored["error_taxonomy"]["counts"] == {
        "transient_false_commit": 1,
        "unknown_false_removal": 1,
    }
    assert restored["final_position_error"]["entries"][0]["error_m"] == 0.0
    assert restored["update_utility_at_0_30_m"]["useful_count"] == 1


def test_action_event_can_be_built_from_contracts() -> None:
    decision = Decision(
        sequence_id="seq",
        track_id="mug",
        frame_id="frame-1",
        timestamp_s=2.0,
        label=LabelClass.PERSISTENT_CHANGE,
        action=DecisionAction.COMMIT,
        confidence=0.9,
        probabilities={
            LabelClass.STABLE: 0.0,
            LabelClass.TRANSIENT: 0.0,
            LabelClass.PERSISTENT_CHANGE: 1.0,
            LabelClass.UNKNOWN: 0.0,
        },
        reason="test",
    )
    update = move(2.0, (1.0, 0.0, 0.0))

    assert ActionEvent.from_record(decision).frame_id == "frame-1"
    assert ActionEvent.from_record(update).new_position_m == (1.0, 0.0, 0.0)
