"""Deterministic error taxonomy for map-maintenance failures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from wise_egomap.contracts import Decision, DecisionAction, EventLabel, MapUpdate
from wise_egomap.evaluation.matching import (
    EventMatching,
    action_in_event,
    match_events,
    normalize_actions,
)
from wise_egomap.evaluation.metrics import (
    is_persistent_change,
    is_transient,
    is_unknown,
)
from wise_egomap.evaluation.models import ActionEvent, ErrorCategory


@dataclass(frozen=True, slots=True)
class TaxonomyEntry:
    """One classified error."""

    category: ErrorCategory
    sequence_id: str
    track_id: str
    timestamp_s: float | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "category": self.category.value,
            "sequence_id": self.sequence_id,
            "track_id": self.track_id,
            "timestamp_s": self.timestamp_s,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class TaxonomyReport:
    """Classified errors and stable category counts."""

    entries: tuple[TaxonomyEntry, ...]
    counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "counts": dict(sorted(self.counts.items())),
        }


def classify_errors(
    events: Iterable[EventLabel],
    actions: Iterable[Decision | MapUpdate | ActionEvent],
    *,
    matching: EventMatching | None = None,
) -> TaxonomyReport:
    """Classify false commits, false removals, and missed changes.

    The taxonomy uses deterministic category order. A commit gets one
    category. If event windows overlap, the precedence is transient, unknown,
    duplicate persistent, and then unmatched.
    """
    event_list = tuple(events)
    action_list = normalize_actions(actions)
    result = matching or match_events(event_list, action_list)
    entries: list[TaxonomyEntry] = []

    for action in result.unmatched_actions:
        transient_events = [
            event
            for event in event_list
            if is_transient(event) and action_in_event(action, event)
        ]
        unknown_events = [
            event
            for event in event_list
            if is_unknown(event) and action_in_event(action, event)
        ]
        persistent_events = [
            event
            for event in event_list
            if is_persistent_change(event) and action_in_event(action, event)
        ]
        if action.action == DecisionAction.COMMIT:
            if transient_events:
                category = ErrorCategory.TRANSIENT_FALSE_COMMIT
                detail = "commit in transient event"
            elif unknown_events:
                category = ErrorCategory.UNKNOWN_EVENT_COMMIT
                detail = "commit in unknown or occlusion event"
            elif persistent_events:
                category = ErrorCategory.DUPLICATE_COMMIT
                detail = "extra commit in persistent-change event"
            else:
                category = ErrorCategory.UNMATCHED_COMMIT
                detail = "commit has no matching event"
            entries.append(
                TaxonomyEntry(
                    category=category,
                    sequence_id=action.sequence_id,
                    track_id=action.track_id,
                    timestamp_s=action.timestamp_s,
                    detail=detail,
                )
            )
        elif action.action == DecisionAction.REJECT and unknown_events:
            entries.append(
                TaxonomyEntry(
                    category=ErrorCategory.UNKNOWN_FALSE_REMOVAL,
                    sequence_id=action.sequence_id,
                    track_id=action.track_id,
                    timestamp_s=action.timestamp_s,
                    detail="reject in unknown or occlusion event",
                )
            )

    matched_persistent = {
        (
            match.event.sequence_id,
            match.event.track_id,
            match.event.start_s,
            match.event.end_s,
        )
        for match in result.matches
        if is_persistent_change(match.event)
        and match.action.action == DecisionAction.COMMIT
    }
    for event in event_list:
        key = (event.sequence_id, event.track_id, event.start_s, event.end_s)
        if is_persistent_change(event) and key not in matched_persistent:
            entries.append(
                TaxonomyEntry(
                    category=ErrorCategory.MISSED_PERSISTENT_CHANGE,
                    sequence_id=event.sequence_id,
                    track_id=event.track_id,
                    timestamp_s=None,
                    detail=(
                        f"no commit in event window [{event.start_s}, {event.end_s}] s"
                    ),
                )
            )

    entries.sort(
        key=lambda entry: (
            entry.category.value,
            entry.sequence_id,
            entry.track_id,
            -1.0 if entry.timestamp_s is None else entry.timestamp_s,
            entry.detail,
        )
    )
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.category.value] = counts.get(entry.category.value, 0) + 1
    return TaxonomyReport(entries=tuple(entries), counts=counts)
