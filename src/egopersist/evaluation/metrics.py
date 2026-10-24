"""Event-level and map-level metrics for EgoPersist."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from egopersist.contracts import (
    Decision,
    DecisionAction,
    EventLabel,
    EventType,
    LabelClass,
    MapUpdate,
)
from egopersist.evaluation.matching import (
    EventMatching,
    action_in_event,
    match_events,
    normalize_actions,
)
from egopersist.evaluation.models import (
    ActionEvent,
    BinaryMetrics,
    PositionReference,
    RateMetric,
    distance_m,
)


def is_persistent_change(event: EventLabel) -> bool:
    """Return true if an event is a persistent change."""
    return (
        event.event_type == EventType.PERSISTENT_CHANGE
        or event.label == LabelClass.PERSISTENT_CHANGE
    )


def is_transient(event: EventLabel) -> bool:
    """Return true if an event is transient."""
    return (
        event.event_type == EventType.TRANSIENT or event.label == LabelClass.TRANSIENT
    )


def is_unknown(event: EventLabel) -> bool:
    """Return true if an event is an unknown or occlusion event."""
    return event.event_type == EventType.OCCLUSION or event.label == LabelClass.UNKNOWN


def _ratio(numerator: int, denominator: int, *, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _binary_metrics(
    true_positives: int,
    predicted_positives: int,
    actual_positives: int,
) -> BinaryMetrics:
    false_positives = predicted_positives - true_positives
    false_negatives = actual_positives - true_positives
    precision = _ratio(true_positives, predicted_positives, empty=0.0)
    recall = _ratio(true_positives, actual_positives, empty=1.0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return BinaryMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        predicted_positives=predicted_positives,
        actual_positives=actual_positives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def persistent_change_metrics(
    events: Iterable[EventLabel],
    actions: Iterable[Decision | MapUpdate | ActionEvent],
    *,
    matching: EventMatching | None = None,
) -> BinaryMetrics:
    """Return persistent-change precision, recall, and F1.

    A true positive is one persistent event matched to one commit. Every
    commit not used by such a match is a false positive. Every persistent
    event without a match is a false negative.
    """
    event_list = tuple(events)
    action_list = normalize_actions(actions)
    result = matching or match_events(event_list, action_list)
    true_positives = sum(
        1
        for match in result.matches
        if is_persistent_change(match.event)
        and match.action.action == DecisionAction.COMMIT
    )
    predicted_positives = sum(
        action.action == DecisionAction.COMMIT for action in action_list
    )
    actual_positives = sum(is_persistent_change(event) for event in event_list)
    return _binary_metrics(true_positives, predicted_positives, actual_positives)


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    """Metrics for one sequence, kept separate from pooled metrics."""

    sequence_id: str
    metrics: BinaryMetrics

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "metrics": self.metrics.to_dict(),
        }


def persistent_change_metrics_by_sequence(
    events: Iterable[EventLabel],
    actions: Iterable[Decision | MapUpdate | ActionEvent],
) -> tuple[SequenceMetrics, ...]:
    """Return persistent-change metrics for each sequence.

    These records are not averaged. Callers must use the pooled function for
    a micro result.
    """
    event_list = tuple(events)
    action_list = normalize_actions(actions)
    sequence_ids = sorted(
        {event.sequence_id for event in event_list}
        | {action.sequence_id for action in action_list}
    )
    records: list[SequenceMetrics] = []
    for sequence_id in sequence_ids:
        sequence_events = tuple(
            event for event in event_list if event.sequence_id == sequence_id
        )
        sequence_actions = tuple(
            action for action in action_list if action.sequence_id == sequence_id
        )
        records.append(
            SequenceMetrics(
                sequence_id=sequence_id,
                metrics=persistent_change_metrics(sequence_events, sequence_actions),
            )
        )
    return tuple(records)


def transient_false_commit_rate(
    events: Iterable[EventLabel],
    actions: Iterable[Decision | MapUpdate | ActionEvent],
) -> RateMetric:
    """Return the fraction of transient events with at least one commit.

    The denominator is the number of transient events. The numerator counts
    an event once, even if it has more than one commit.
    """
    event_list = tuple(events)
    action_list = normalize_actions(actions)
    transient_events = tuple(event for event in event_list if is_transient(event))
    affected = sum(
        any(
            action.action == DecisionAction.COMMIT and action_in_event(action, event)
            for action in action_list
        )
        for event in transient_events
    )
    return RateMetric(
        numerator=affected,
        denominator=len(transient_events),
        rate=_ratio(affected, len(transient_events), empty=0.0),
    )


def unknown_false_removal_rate(
    events: Iterable[EventLabel],
    actions: Iterable[Decision | MapUpdate | ActionEvent],
    *,
    removal_actions: Sequence[DecisionAction] = (DecisionAction.REJECT,),
) -> RateMetric:
    """Return the fraction of unknown events with a false removal.

    Unknown events are occlusion events or events with the UNKNOWN label. A
    removal is an action in ``removal_actions``. The default is REJECT.
    """
    event_list = tuple(events)
    action_list = normalize_actions(actions)
    unknown_events = tuple(event for event in event_list if is_unknown(event))
    removal_set = set(removal_actions)
    affected = sum(
        any(
            action.action in removal_set and action_in_event(action, event)
            for action in action_list
        )
        for event in unknown_events
    )
    return RateMetric(
        numerator=affected,
        denominator=len(unknown_events),
        rate=_ratio(affected, len(unknown_events), empty=0.0),
    )


@dataclass(frozen=True, slots=True)
class CommitTimingEntry:
    """Commit timing for one persistent event."""

    sequence_id: str
    track_id: str
    event_start_s: float
    event_end_s: float
    time_to_correct_commit_s: float | None
    stale_map_duration_s: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "track_id": self.track_id,
            "event_start_s": self.event_start_s,
            "event_end_s": self.event_end_s,
            "time_to_correct_commit_s": self.time_to_correct_commit_s,
            "stale_map_duration_s": self.stale_map_duration_s,
        }


@dataclass(frozen=True, slots=True)
class CommitTimingReport:
    """Aggregate and per-event commit timing."""

    entries: tuple[CommitTimingEntry, ...]
    time_to_correct_commit_s: float | None
    total_stale_map_duration_s: float
    mean_stale_map_duration_s: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "time_to_correct_commit_s": self.time_to_correct_commit_s,
            "total_stale_map_duration_s": self.total_stale_map_duration_s,
            "mean_stale_map_duration_s": self.mean_stale_map_duration_s,
        }


def commit_timing(
    events: Iterable[EventLabel],
    actions: Iterable[Decision | MapUpdate | ActionEvent],
) -> CommitTimingReport:
    """Measure time to first commit and stale-map duration.

    The clock starts at the event start. The first commit in the closed event
    window is a correct commit. If no commit occurs, the stale duration is the
    full event duration and time to correct commit is undefined (None).
    """
    event_list = tuple(events)
    action_list = normalize_actions(actions)
    entries: list[CommitTimingEntry] = []
    for event in sorted(
        (event for event in event_list if is_persistent_change(event)),
        key=lambda item: (
            item.sequence_id,
            item.track_id,
            item.start_s,
            item.end_s,
            item.event_type.value,
        ),
    ):
        commits = sorted(
            action.timestamp_s
            for action in action_list
            if action.action == DecisionAction.COMMIT and action_in_event(action, event)
        )
        delay = commits[0] - event.start_s if commits else None
        stale = delay if delay is not None else event.end_s - event.start_s
        entries.append(
            CommitTimingEntry(
                sequence_id=event.sequence_id,
                track_id=event.track_id,
                event_start_s=event.start_s,
                event_end_s=event.end_s,
                time_to_correct_commit_s=delay,
                stale_map_duration_s=stale,
            )
        )
    delays: list[float] = [
        entry.time_to_correct_commit_s
        for entry in entries
        if entry.time_to_correct_commit_s is not None
    ]
    total_stale = sum(entry.stale_map_duration_s for entry in entries)
    return CommitTimingReport(
        entries=tuple(entries),
        time_to_correct_commit_s=(sum(delays) / len(delays) if delays else None),
        total_stale_map_duration_s=total_stale,
        mean_stale_map_duration_s=(total_stale / len(entries) if entries else 0.0),
    )


@dataclass(frozen=True, slots=True)
class FinalPositionErrorEntry:
    """Final map position error for one track."""

    sequence_id: str
    track_id: str
    reference_timestamp_s: float
    error_m: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "track_id": self.track_id,
            "reference_timestamp_s": self.reference_timestamp_s,
            "error_m": self.error_m,
        }


@dataclass(frozen=True, slots=True)
class FinalPositionErrorReport:
    """Final map position errors."""

    entries: tuple[FinalPositionErrorEntry, ...]
    mean_error_m: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "mean_error_m": self.mean_error_m,
        }


def final_position_error_m(
    updates: Iterable[MapUpdate],
    references: Iterable[PositionReference],
) -> FinalPositionErrorReport:
    """Return the final causal map-position error for each track.

    The evaluator keeps the last reference for each sequence-track pair. It
    uses the last map update at or before the reference timestamp. If no such
    update exists, the error is None and the track does not enter the mean.
    """
    update_list = sorted(
        updates,
        key=lambda update: (
            update.sequence_id,
            update.track_id,
            update.timestamp_s,
            update.map_version,
        ),
    )
    final_references: dict[tuple[str, str], PositionReference] = {}
    for reference in references:
        key = (reference.sequence_id, reference.track_id)
        current = final_references.get(key)
        if current is None or reference.timestamp_s > current.timestamp_s:
            final_references[key] = reference
    entries: list[FinalPositionErrorEntry] = []
    for key in sorted(final_references):
        reference = final_references[key]
        usable = [
            update
            for update in update_list
            if (update.sequence_id, update.track_id) == key
            and update.timestamp_s <= reference.timestamp_s
            and update.new_position_m is not None
        ]
        last_position = usable[-1].new_position_m if usable else None
        error = (
            distance_m(last_position, reference.position_m)
            if last_position is not None
            else None
        )
        entries.append(
            FinalPositionErrorEntry(
                sequence_id=reference.sequence_id,
                track_id=reference.track_id,
                reference_timestamp_s=reference.timestamp_s,
                error_m=error,
            )
        )
    errors = [entry.error_m for entry in entries if entry.error_m is not None]
    return FinalPositionErrorReport(
        entries=tuple(entries),
        mean_error_m=(
            sum(error for error in errors if error is not None) / len(errors)
            if errors
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class UpdateUtilityEntry:
    """Utility of one commit at the position tolerance."""

    sequence_id: str
    track_id: str
    timestamp_s: float
    previous_error_m: float
    new_error_m: float
    utility: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "track_id": self.track_id,
            "timestamp_s": self.timestamp_s,
            "previous_error_m": self.previous_error_m,
            "new_error_m": self.new_error_m,
            "utility": self.utility,
        }


@dataclass(frozen=True, slots=True)
class UpdateUtilityReport:
    """Update utility at a position tolerance."""

    threshold_m: float
    entries: tuple[UpdateUtilityEntry, ...]
    mean_utility: float
    useful_count: int
    harmful_count: int
    neutral_count: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary."""
        return {
            "threshold_m": self.threshold_m,
            "entries": [entry.to_dict() for entry in self.entries],
            "mean_utility": self.mean_utility,
            "useful_count": self.useful_count,
            "harmful_count": self.harmful_count,
            "neutral_count": self.neutral_count,
        }


def latest_reference(
    references: Sequence[PositionReference],
    *,
    sequence_id: str,
    track_id: str,
    timestamp_s: float,
) -> PositionReference | None:
    """Return the latest causal reference for one track."""
    causal = [
        reference
        for reference in references
        if reference.sequence_id == sequence_id
        and reference.track_id == track_id
        and reference.timestamp_s <= timestamp_s
    ]
    return max(causal, key=lambda item: item.timestamp_s) if causal else None


def update_utility_at_0_30_m(
    updates: Iterable[MapUpdate],
    references: Iterable[PositionReference],
    *,
    threshold_m: float = 0.30,
) -> UpdateUtilityReport:
    """Return mean decision utility for commits at a 0.30 m tolerance.

    For each commit with a causal reference and a previous position:

    - utility is +1.0 if the previous error is above the tolerance and the new
      error is at or below it;
    - utility is -1.0 if the previous error is at or below the tolerance and
      the new error is above it;
    - utility is 0.0 in all other cases.

    The mean uses scorable commits only. With no scorable commits, it is 0.0.
    """
    if threshold_m < 0.0:
        raise ValueError("threshold_m must be non-negative")
    reference_list = tuple(
        sorted(
            references,
            key=lambda item: (
                item.sequence_id,
                item.track_id,
                item.timestamp_s,
                item.position_m,
            ),
        )
    )
    entries: list[UpdateUtilityEntry] = []
    for update in sorted(
        updates,
        key=lambda item: (
            item.sequence_id,
            item.track_id,
            item.timestamp_s,
            item.map_version,
        ),
    ):
        if (
            update.action != DecisionAction.COMMIT
            or update.previous_position_m is None
            or update.new_position_m is None
        ):
            continue
        reference = latest_reference(
            reference_list,
            sequence_id=update.sequence_id,
            track_id=update.track_id,
            timestamp_s=update.timestamp_s,
        )
        if reference is None:
            continue
        previous_error = distance_m(update.previous_position_m, reference.position_m)
        new_error = distance_m(update.new_position_m, reference.position_m)
        if previous_error > threshold_m >= new_error:
            utility = 1.0
        elif previous_error <= threshold_m < new_error:
            utility = -1.0
        else:
            utility = 0.0
        entries.append(
            UpdateUtilityEntry(
                sequence_id=update.sequence_id,
                track_id=update.track_id,
                timestamp_s=update.timestamp_s,
                previous_error_m=previous_error,
                new_error_m=new_error,
                utility=utility,
            )
        )
    useful = sum(entry.utility > 0.0 for entry in entries)
    harmful = sum(entry.utility < 0.0 for entry in entries)
    return UpdateUtilityReport(
        threshold_m=threshold_m,
        entries=tuple(entries),
        mean_utility=(
            sum(entry.utility for entry in entries) / len(entries) if entries else 0.0
        ),
        useful_count=useful,
        harmful_count=harmful,
        neutral_count=len(entries) - useful - harmful,
    )


def update_utility_at_030_m(
    updates: Iterable[MapUpdate],
    references: Iterable[PositionReference],
) -> UpdateUtilityReport:
    """Alias for update utility at the fixed 0.30 m tolerance."""
    return update_utility_at_0_30_m(updates, references, threshold_m=0.30)
