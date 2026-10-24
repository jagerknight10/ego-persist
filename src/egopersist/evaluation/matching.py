"""Deterministic one-to-one matching between labels and map actions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from egopersist.contracts import (
    Decision,
    DecisionAction,
    EventLabel,
    EventType,
    MapUpdate,
)
from egopersist.evaluation.models import ActionEvent

_EXPECTED_ACTION: dict[EventType, DecisionAction] = {
    EventType.STABLE: DecisionAction.KEEP,
    EventType.TRANSIENT: DecisionAction.REJECT,
    EventType.PERSISTENT_CHANGE: DecisionAction.COMMIT,
    EventType.OCCLUSION: DecisionAction.DEFER,
    EventType.REVERSION: DecisionAction.REVERT,
}


def expected_action(event_type: EventType) -> DecisionAction:
    """Return the map action required by an event type."""
    return _EXPECTED_ACTION[event_type]


def _event_key(event: EventLabel) -> tuple[str, str, float, float, str, str, str]:
    return (
        event.sequence_id,
        event.track_id,
        event.start_s,
        event.end_s,
        event.event_type.value,
        event.label.value,
        event.source,
    )


def _action_sort_key(
    action: ActionEvent,
) -> tuple[str, str, float, str, str, tuple[float, ...]]:
    position = action.new_position_m or ()
    return (
        action.sequence_id,
        action.track_id,
        action.timestamp_s,
        action.action.value,
        action.frame_id,
        position,
    )


def normalize_actions(
    actions: Iterable[Decision | MapUpdate | ActionEvent],
) -> tuple[ActionEvent, ...]:
    """Convert decisions, map updates, and action events to action events."""
    normalized = [
        action if isinstance(action, ActionEvent) else ActionEvent.from_record(action)
        for action in actions
    ]
    return tuple(sorted(normalized, key=_action_sort_key))


def action_in_event(action: ActionEvent, event: EventLabel) -> bool:
    """Return true if an action is in the event window for the same track."""
    return (
        action.sequence_id == event.sequence_id
        and action.track_id == event.track_id
        and event.start_s <= action.timestamp_s <= event.end_s
    )


@dataclass(frozen=True, slots=True)
class EventMatch:
    """One event matched to one action."""

    event: EventLabel
    action: ActionEvent

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {"event": self.event.to_dict(), "action": self.action.to_dict()}


@dataclass(frozen=True, slots=True)
class EventMatching:
    """Result of deterministic one-to-one event matching."""

    matches: tuple[EventMatch, ...]
    unmatched_events: tuple[EventLabel, ...]
    unmatched_actions: tuple[ActionEvent, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "matched": [match.to_dict() for match in self.matches],
            "unmatched_events": [event.to_dict() for event in self.unmatched_events],
            "unmatched_actions": [
                action.to_dict() for action in self.unmatched_actions
            ],
        }


def match_events(
    events: Iterable[EventLabel],
    actions: Iterable[Decision | MapUpdate | ActionEvent],
) -> EventMatching:
    """Match events to actions with maximum cardinality and stable ordering.

    A candidate pair has the same sequence and track, the action is in the
    closed event window, and the action equals the event's expected action.
    Each event and each action can occur in at most one match.
    """
    event_list = tuple(sorted(events, key=_event_key))
    action_list = normalize_actions(actions)

    candidates: list[list[int]] = []
    for event in event_list:
        required = expected_action(event.event_type)
        candidates.append(
            [
                index
                for index, action in enumerate(action_list)
                if action.action == required and action_in_event(action, event)
            ]
        )

    action_owner: dict[int, int] = {}
    event_action: dict[int, int] = {}

    def augment(event_index: int, visited: set[int]) -> bool:
        for action_index in candidates[event_index]:
            if action_index in visited:
                continue
            visited.add(action_index)
            owner = action_owner.get(action_index)
            if owner is None or augment(owner, visited):
                action_owner[action_index] = event_index
                event_action[event_index] = action_index
                return True
        return False

    for event_index in range(len(event_list)):
        augment(event_index, set())

    matches = tuple(
        EventMatch(event_list[event_index], action_list[action_index])
        for event_index, action_index in sorted(
            event_action.items(),
            key=lambda item: _event_key(event_list[item[0]]),
        )
    )
    matched_event_indices = set(event_action)
    matched_action_indices = set(event_action.values())
    unmatched_events = tuple(
        event
        for index, event in enumerate(event_list)
        if index not in matched_event_indices
    )
    unmatched_actions = tuple(
        action
        for index, action in enumerate(action_list)
        if index not in matched_action_indices
    )
    return EventMatching(matches, unmatched_events, unmatched_actions)
