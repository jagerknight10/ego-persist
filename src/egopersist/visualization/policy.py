"""Small policy adapters for the synthetic replay.

`DemoStateMachineAdapter` is the integration seam. A future mapping module can
replace it without a change to the replay, viewer, query, or renderer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from egopersist.contracts import Decision, DecisionAction, LabelClass
from egopersist.visualization.replay import (
    Position3,
    ReplayEvidence,
    ReplayMapEntry,
    ReplayObservation,
)

CHANGE_TRANSLATION_M = 0.10
COMMIT_MIN_KEYFRAMES = 5
COMMIT_MIN_SPAN_S = 5.0
MIN_CO_VISIBILITY = 0.35
MAP_BOOTSTRAP_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """Output of one causal policy step."""

    map_entries: tuple[ReplayMapEntry, ...]
    decisions: tuple[Decision, ...]
    evidence: tuple[ReplayEvidence, ...]


class StateMachineAdapter(Protocol):
    """Adapter contract used by the demo and a later state machine."""

    def reset(self, sequence_id: str, initial_map: Sequence[ReplayMapEntry]) -> None:
        """Remove state and load a committed map."""

    def update(
        self,
        frame_id: str,
        timestamp_s: float,
        observations: Sequence[ReplayObservation],
    ) -> PolicySnapshot:
        """Process one keyframe. Do not read later frames."""


@dataclass(slots=True)
class _Track:
    object_id: str
    display_name: str
    kind: str
    position_m: Position3
    state: LabelClass = LabelClass.STABLE
    confidence: float = MAP_BOOTSTRAP_CONFIDENCE
    candidate_position_m: Position3 | None = None
    candidate_first_s: float = 0.0
    candidate_count: int = 0
    missing_count: int = 0


def _probabilities(
    *,
    stable: float,
    transient: float,
    persistent_change: float,
    unknown: float,
) -> dict[LabelClass, float]:
    values = {
        LabelClass.STABLE: stable,
        LabelClass.TRANSIENT: transient,
        LabelClass.PERSISTENT_CHANGE: persistent_change,
        LabelClass.UNKNOWN: unknown,
    }
    total = sum(values.values())
    return {label: value / total for label, value in values.items()}


def _distance(first: Position3, second: Position3) -> float:
    return (
        sum((left - right) ** 2 for left, right in zip(first, second, strict=True))
        ** 0.5
    )


def _evidence(
    object_id: str,
    *,
    name: str,
    label: str,
    value: float,
    unit: str,
    threshold: float,
    max_value: float,
) -> ReplayEvidence:
    return ReplayEvidence(
        object_id=object_id,
        name=name,
        label=label,
        value=min(value, max_value),
        unit=unit,
        threshold=threshold,
        max_value=max_value,
    )


def _stable_evidence(
    object_id: str,
    *,
    displacement_m: float,
    confidence: float,
) -> tuple[ReplayEvidence, ...]:
    return (
        _evidence(
            object_id,
            name="displacement",
            label="Pose delta",
            value=displacement_m,
            unit="m",
            threshold=CHANGE_TRANSLATION_M,
            max_value=1.0,
        ),
        _evidence(
            object_id,
            name="observation_support",
            label="Mask and depth support",
            value=confidence,
            unit="score",
            threshold=MIN_CO_VISIBILITY,
            max_value=1.0,
        ),
    )


def _decision(
    *,
    sequence_id: str,
    frame_id: str,
    timestamp_s: float,
    object_id: str,
    label: LabelClass,
    action: DecisionAction,
    confidence: float,
    probabilities: dict[LabelClass, float],
    reason: str,
) -> Decision:
    return Decision(
        sequence_id=sequence_id,
        track_id=object_id,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        label=label,
        action=action,
        confidence=confidence,
        probabilities=probabilities,
        reason=reason,
    )


def _map_entry(track: _Track, *, map_version: int) -> ReplayMapEntry:
    return ReplayMapEntry(
        object_id=track.object_id,
        display_name=track.display_name,
        kind=track.kind,
        position_m=track.position_m,
        state=track.state,
        confidence=track.confidence,
        map_version=map_version,
    )


class DemoStateMachineAdapter:
    """Causal policy that applies persistence gates before a map commit."""

    def __init__(self) -> None:
        self._sequence_id = ""
        self._tracks: dict[str, _Track] = {}
        self._map_version = 0

    def reset(self, sequence_id: str, initial_map: Sequence[ReplayMapEntry]) -> None:
        """Load the committed map for a new sequence."""
        if not sequence_id:
            raise ValueError("sequence_id must be non-empty")
        self._sequence_id = sequence_id
        self._tracks = {
            item.object_id: _Track(
                object_id=item.object_id,
                display_name=item.display_name,
                kind=item.kind,
                position_m=item.position_m,
                state=LabelClass.STABLE,
                confidence=MAP_BOOTSTRAP_CONFIDENCE,
            )
            for item in initial_map
        }
        self._map_version = max((item.map_version for item in initial_map), default=0)

    def update(
        self,
        frame_id: str,
        timestamp_s: float,
        observations: Sequence[ReplayObservation],
    ) -> PolicySnapshot:
        """Apply a persistence gate to one causal keyframe."""
        if not self._sequence_id:
            raise RuntimeError("reset must be called before update")
        if not frame_id:
            raise ValueError("frame_id must be non-empty")
        decisions: list[Decision] = []
        evidence: list[ReplayEvidence] = []
        observed_ids = {item.object_id for item in observations}

        for observation in observations:
            track = self._tracks.get(observation.object_id)
            if track is None:
                track = self._new_track(observation)
                if track is None:
                    decisions.append(
                        _decision(
                            sequence_id=self._sequence_id,
                            frame_id=frame_id,
                            timestamp_s=timestamp_s,
                            object_id=observation.object_id,
                            label=LabelClass.TRANSIENT,
                            action=DecisionAction.REJECT,
                            confidence=0.97,
                            probabilities=_probabilities(
                                stable=0.01,
                                transient=0.97,
                                persistent_change=0.01,
                                unknown=0.01,
                            ),
                            reason="Human-caused short presence. Reject the candidate.",
                        )
                    )
                    evidence.append(
                        _evidence(
                            observation.object_id,
                            name="human_cause",
                            label="Human entity",
                            value=1.0,
                            unit="flag",
                            threshold=1.0,
                            max_value=1.0,
                        )
                    )
                    evidence.append(
                        _evidence(
                            observation.object_id,
                            name="map_support",
                            label="Committed map support",
                            value=0.0,
                            unit="keyframes",
                            threshold=COMMIT_MIN_KEYFRAMES,
                            max_value=float(COMMIT_MIN_KEYFRAMES),
                        )
                    )
                    continue

            if observation.confidence < MIN_CO_VISIBILITY:
                track.missing_count += 1
                track.state = LabelClass.UNKNOWN
                track.confidence = max(0.35, track.confidence * 0.9)
                decisions.append(
                    _decision(
                        sequence_id=self._sequence_id,
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        object_id=observation.object_id,
                        label=LabelClass.UNKNOWN,
                        action=DecisionAction.DEFER,
                        confidence=0.90,
                        probabilities=_probabilities(
                            stable=0.02,
                            transient=0.02,
                            persistent_change=0.02,
                            unknown=0.94,
                        ),
                        reason="Weak visibility. Keep the map entry and defer.",
                    )
                )
                evidence.append(
                    _evidence(
                        observation.object_id,
                        name="visibility",
                        label="Mask and depth support",
                        value=observation.confidence,
                        unit="score",
                        threshold=MIN_CO_VISIBILITY,
                        max_value=1.0,
                    )
                )
                continue

            displacement_m = _distance(track.position_m, observation.position_m)
            if displacement_m <= CHANGE_TRANSLATION_M:
                track.state = LabelClass.STABLE
                track.confidence = 0.98
                track.candidate_position_m = None
                track.candidate_first_s = 0.0
                track.candidate_count = 0
                track.missing_count = 0
                decisions.append(
                    _decision(
                        sequence_id=self._sequence_id,
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        object_id=observation.object_id,
                        label=LabelClass.STABLE,
                        action=DecisionAction.KEEP,
                        confidence=0.98,
                        probabilities=_probabilities(
                            stable=0.98,
                            transient=0.01,
                            persistent_change=0.005,
                            unknown=0.005,
                        ),
                        reason="Observation agrees with the committed map.",
                    )
                )
                evidence.extend(
                    _stable_evidence(
                        observation.object_id,
                        displacement_m=displacement_m,
                        confidence=observation.confidence,
                    )
                )
                continue

            if (
                track.candidate_position_m is None
                or _distance(track.candidate_position_m, observation.position_m) > 0.15
            ):
                track.candidate_position_m = observation.position_m
                track.candidate_first_s = timestamp_s
                track.candidate_count = 1
            else:
                track.candidate_count += 1
                track.candidate_position_m = observation.position_m

            span_s = timestamp_s - track.candidate_first_s
            can_commit = (
                track.candidate_count >= COMMIT_MIN_KEYFRAMES
                and span_s >= COMMIT_MIN_SPAN_S
                and observation.confidence >= MIN_CO_VISIBILITY
            )
            evidence.extend(
                (
                    _evidence(
                        observation.object_id,
                        name="candidate_keyframes",
                        label="Visible changed keyframes",
                        value=float(track.candidate_count),
                        unit="keyframes",
                        threshold=float(COMMIT_MIN_KEYFRAMES),
                        max_value=15.0,
                    ),
                    _evidence(
                        observation.object_id,
                        name="candidate_dwell",
                        label="Candidate dwell time",
                        value=span_s,
                        unit="s",
                        threshold=COMMIT_MIN_SPAN_S,
                        max_value=7.0,
                    ),
                    _evidence(
                        observation.object_id,
                        name="displacement",
                        label="Pose delta",
                        value=displacement_m,
                        unit="m",
                        threshold=CHANGE_TRANSLATION_M,
                        max_value=1.0,
                    ),
                )
            )
            if can_commit:
                self._map_version += 1
                track.position_m = observation.position_m
                track.state = LabelClass.STABLE
                track.confidence = 0.96
                track.candidate_position_m = None
                track.candidate_first_s = 0.0
                track.candidate_count = 0
                track.missing_count = 0
                decisions.append(
                    _decision(
                        sequence_id=self._sequence_id,
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        object_id=observation.object_id,
                        label=LabelClass.PERSISTENT_CHANGE,
                        action=DecisionAction.COMMIT,
                        confidence=0.96,
                        probabilities=_probabilities(
                            stable=0.01,
                            transient=0.01,
                            persistent_change=0.97,
                            unknown=0.01,
                        ),
                        reason=(
                            f"Gate passed: {track.candidate_count} visible keyframes, "
                            f"{span_s:.1f} s dwell."
                        ),
                    )
                )
            else:
                track.state = LabelClass.UNKNOWN
                track.confidence = 0.70
                decisions.append(
                    _decision(
                        sequence_id=self._sequence_id,
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        object_id=observation.object_id,
                        label=LabelClass.UNKNOWN,
                        action=DecisionAction.DEFER,
                        confidence=0.72,
                        probabilities=_probabilities(
                            stable=0.03,
                            transient=0.03,
                            persistent_change=0.84,
                            unknown=0.10,
                        ),
                        reason=(
                            "Changed pose has support, but the persistence gate "
                            "is not complete."
                        ),
                    )
                )

        for object_id, track in self._tracks.items():
            if object_id in observed_ids:
                continue
            track.missing_count += 1
            track.state = LabelClass.UNKNOWN
            track.confidence = max(0.30, track.confidence - 0.05)
            decisions.append(
                _decision(
                    sequence_id=self._sequence_id,
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    object_id=object_id,
                    label=LabelClass.UNKNOWN,
                    action=DecisionAction.DEFER,
                    confidence=0.88,
                    probabilities=_probabilities(
                        stable=0.02,
                        transient=0.03,
                        persistent_change=0.02,
                        unknown=0.93,
                    ),
                    reason=(
                        f"No reliable observation for {track.missing_count} keyframes. "
                        "Keep the map entry."
                    ),
                )
            )
            evidence.extend(
                (
                    _evidence(
                        object_id,
                        name="visibility",
                        label="Current visibility",
                        value=0.0,
                        unit="score",
                        threshold=MIN_CO_VISIBILITY,
                        max_value=1.0,
                    ),
                    _evidence(
                        object_id,
                        name="retained_map",
                        label="Map retention",
                        value=1.0,
                        unit="flag",
                        threshold=1.0,
                        max_value=1.0,
                    ),
                )
            )

        entries = tuple(
            _map_entry(track, map_version=self._map_version)
            for track in sorted(self._tracks.values(), key=lambda item: item.object_id)
        )
        return PolicySnapshot(
            map_entries=entries,
            decisions=tuple(
                sorted(decisions, key=lambda item: (item.track_id, item.frame_id))
            ),
            evidence=tuple(
                sorted(
                    evidence,
                    key=lambda item: (
                        item.object_id,
                        item.name,
                        item.value,
                        item.label,
                    ),
                )
            ),
        )

    def _new_track(self, observation: ReplayObservation) -> _Track | None:
        if observation.human_caused:
            return None
        if observation.confidence < MIN_CO_VISIBILITY:
            return None
        track = _Track(
            object_id=observation.object_id,
            display_name=observation.display_name,
            kind=observation.kind,
            position_m=observation.position_m,
            state=LabelClass.UNKNOWN,
            confidence=0.70,
        )
        self._tracks[observation.object_id] = track
        return track


class InstantUpdateAdapter:
    """Naive comparison policy that applies every observation at once."""

    def __init__(self) -> None:
        self._sequence_id = ""
        self._objects: dict[str, ReplayMapEntry] = {}
        self._map_version = 0

    def reset(self, sequence_id: str, initial_map: Sequence[ReplayMapEntry]) -> None:
        """Load the map for a new sequence."""
        if not sequence_id:
            raise ValueError("sequence_id must be non-empty")
        self._sequence_id = sequence_id
        self._objects = {
            item.object_id: ReplayMapEntry(
                object_id=item.object_id,
                display_name=item.display_name,
                kind=item.kind,
                position_m=item.position_m,
                state=LabelClass.STABLE,
                confidence=MAP_BOOTSTRAP_CONFIDENCE,
                map_version=item.map_version,
            )
            for item in initial_map
        }
        self._map_version = max((item.map_version for item in initial_map), default=0)

    def update(
        self,
        frame_id: str,
        timestamp_s: float,
        observations: Sequence[ReplayObservation],
    ) -> PolicySnapshot:
        """Replace the map with the current observations."""
        if not self._sequence_id:
            raise RuntimeError("reset must be called before update")
        observed_ids = {item.object_id for item in observations}
        decisions: list[Decision] = []
        evidence: list[ReplayEvidence] = []
        self._objects = {
            object_id: item
            for object_id, item in self._objects.items()
            if object_id in observed_ids
        }
        for observation in observations:
            previous = self._objects.get(observation.object_id)
            changed = (
                previous is None
                or _distance(previous.position_m, observation.position_m)
                > CHANGE_TRANSLATION_M
            )
            if changed:
                self._map_version += 1
            self._objects[observation.object_id] = ReplayMapEntry(
                object_id=observation.object_id,
                display_name=observation.display_name,
                kind=observation.kind,
                position_m=observation.position_m,
                state=LabelClass.STABLE,
                confidence=observation.confidence,
                map_version=self._map_version,
            )
            decisions.append(
                _decision(
                    sequence_id=self._sequence_id,
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    object_id=observation.object_id,
                    label=(
                        LabelClass.PERSISTENT_CHANGE if changed else LabelClass.STABLE
                    ),
                    action=(DecisionAction.COMMIT if changed else DecisionAction.KEEP),
                    confidence=0.99,
                    probabilities=_probabilities(
                        stable=0.01 if changed else 0.98,
                        transient=0.005,
                        persistent_change=0.98 if changed else 0.01,
                        unknown=0.005,
                    ),
                    reason=(
                        "Instant update uses the current observation directly."
                        if changed
                        else "Observation matches the current map."
                    ),
                )
            )
            evidence.append(
                _evidence(
                    observation.object_id,
                    name="instant_support",
                    label="Current observation",
                    value=observation.confidence,
                    unit="score",
                    threshold=0.0,
                    max_value=1.0,
                )
            )
        entries = tuple(sorted(self._objects.values(), key=lambda item: item.object_id))
        return PolicySnapshot(
            map_entries=entries,
            decisions=tuple(decisions),
            evidence=tuple(evidence),
        )
