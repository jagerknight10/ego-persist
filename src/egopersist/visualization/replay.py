"""Deterministic replay records for the synthetic EgoPersist demo.

The schema contains only serializable values. It shows the observations that
the policies receive, the resulting maps, the decisions, the evidence, and the
mug query. It does not contain ground-truth object pose as a policy input.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar

from egopersist.contracts import DecisionAction, LabelClass

SCHEMA_VERSION = "egopersist.demo.replay/1"

Position3 = tuple[float, float, float]


class ReplaySchemaError(ValueError):
    """Raised when replay data does not match the stable schema."""


def _mapping(
    value: Any,
    *,
    name: str,
    fields: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplaySchemaError(f"{name} must be an object")
    keys = set(value)
    unknown = keys - fields
    missing = fields - keys
    if unknown:
        raise ReplaySchemaError(f"{name} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ReplaySchemaError(f"{name} has missing fields: {sorted(missing)}")
    if any(not isinstance(key, str) for key in value):
        raise ReplaySchemaError(f"{name} keys must be strings")
    return {str(key): item for key, item in value.items()}


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplaySchemaError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplaySchemaError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ReplaySchemaError(f"{name} must be finite")
    return result


def _non_negative(value: Any, *, name: str) -> float:
    result = _number(value, name=name)
    if result < 0.0:
        raise ReplaySchemaError(f"{name} must be non-negative")
    return result


def _ratio(value: Any, *, name: str) -> float:
    result = _number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ReplaySchemaError(f"{name} must be in [0, 1]")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReplaySchemaError(f"{name} must be a non-negative integer")
    return value


def _position(value: Any, *, name: str) -> Position3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ReplaySchemaError(f"{name} must have three values")
    return (
        _number(value[0], name=f"{name}[0]"),
        _number(value[1], name=f"{name}[1]"),
        _number(value[2], name=f"{name}[2]"),
    )


def _label(value: Any, *, name: str) -> LabelClass:
    try:
        return LabelClass(value)
    except (TypeError, ValueError) as exc:
        raise ReplaySchemaError(f"{name} is not a valid label") from exc


def _action(value: Any, *, name: str) -> DecisionAction:
    try:
        return DecisionAction(value)
    except (TypeError, ValueError) as exc:
        raise ReplaySchemaError(f"{name} is not a valid action") from exc


def _sequence(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReplaySchemaError(f"{name} must be a list")
    return value


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    """One visible synthetic observation given to both policies."""

    object_id: str
    display_name: str
    kind: str
    position_m: Position3
    extent_m: Position3
    confidence: float
    human_caused: bool

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "object_id",
            "display_name",
            "kind",
            "position_m",
            "extent_m",
            "confidence",
            "human_caused",
        }
    )

    def __post_init__(self) -> None:
        for name in ("object_id", "display_name", "kind"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ReplaySchemaError(f"{name} must be a non-empty string")
        object.__setattr__(
            self, "position_m", _position(self.position_m, name="position_m")
        )
        extent = _position(self.extent_m, name="extent_m")
        if any(item <= 0.0 for item in extent):
            raise ReplaySchemaError("extent_m values must be positive")
        object.__setattr__(self, "extent_m", extent)
        object.__setattr__(self, "confidence", _ratio(self.confidence, name="confidence"))
        if not isinstance(self.human_caused, bool):
            raise ReplaySchemaError("human_caused must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "object_id": self.object_id,
            "display_name": self.display_name,
            "kind": self.kind,
            "position_m": list(self.position_m),
            "extent_m": list(self.extent_m),
            "confidence": self.confidence,
            "human_caused": self.human_caused,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayObservation:
        """Build and validate one observation."""
        item = _mapping(data, name="observation", fields=cls._FIELDS)
        human_caused = item["human_caused"]
        if not isinstance(human_caused, bool):
            raise ReplaySchemaError("human_caused must be a boolean")
        return cls(
            object_id=_text(item["object_id"], name="object_id"),
            display_name=_text(item["display_name"], name="display_name"),
            kind=_text(item["kind"], name="kind"),
            position_m=_position(item["position_m"], name="position_m"),
            extent_m=_position(item["extent_m"], name="extent_m"),
            confidence=_ratio(item["confidence"], name="confidence"),
            human_caused=human_caused,
        )


@dataclass(frozen=True, slots=True)
class ReplayMapEntry:
    """One object in a policy map."""

    object_id: str
    display_name: str
    kind: str
    position_m: Position3
    state: LabelClass
    confidence: float
    map_version: int

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "object_id",
            "display_name",
            "kind",
            "position_m",
            "state",
            "confidence",
            "map_version",
        }
    )

    def __post_init__(self) -> None:
        for name in ("object_id", "display_name", "kind"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ReplaySchemaError(f"{name} must be a non-empty string")
        object.__setattr__(
            self, "position_m", _position(self.position_m, name="position_m")
        )
        if not isinstance(self.state, LabelClass):
            raise ReplaySchemaError("state must be a LabelClass")
        object.__setattr__(
            self, "confidence", _ratio(self.confidence, name="confidence")
        )
        if (
            not isinstance(self.map_version, int)
            or isinstance(self.map_version, bool)
            or self.map_version < 0
        ):
            raise ReplaySchemaError("map_version must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "object_id": self.object_id,
            "display_name": self.display_name,
            "kind": self.kind,
            "position_m": list(self.position_m),
            "state": self.state.value,
            "confidence": self.confidence,
            "map_version": self.map_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayMapEntry:
        """Build and validate one map entry."""
        item = _mapping(data, name="map_entry", fields=cls._FIELDS)
        return cls(
            object_id=_text(item["object_id"], name="object_id"),
            display_name=_text(item["display_name"], name="display_name"),
            kind=_text(item["kind"], name="kind"),
            position_m=_position(item["position_m"], name="position_m"),
            state=_label(item["state"], name="state"),
            confidence=_ratio(item["confidence"], name="confidence"),
            map_version=_integer(item["map_version"], name="map_version"),
        )


@dataclass(frozen=True, slots=True)
class ReplayEvidence:
    """One evidence reading that explains a decision."""

    object_id: str
    name: str
    label: str
    value: float
    unit: str
    threshold: float
    max_value: float

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "object_id",
            "name",
            "label",
            "value",
            "unit",
            "threshold",
            "max_value",
        }
    )

    def __post_init__(self) -> None:
        for name in ("object_id", "name", "label", "unit"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ReplaySchemaError(f"{name} must be a non-empty string")
        object.__setattr__(self, "value", _non_negative(self.value, name="value"))
        object.__setattr__(
            self, "threshold", _non_negative(self.threshold, name="threshold")
        )
        object.__setattr__(
            self, "max_value", _non_negative(self.max_value, name="max_value")
        )
        if self.max_value <= 0.0:
            raise ReplaySchemaError("max_value must be positive")
        if self.value > self.max_value:
            raise ReplaySchemaError("value must not exceed max_value")
        if self.threshold > self.max_value:
            raise ReplaySchemaError("threshold must not exceed max_value")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "object_id": self.object_id,
            "name": self.name,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "threshold": self.threshold,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayEvidence:
        """Build and validate one evidence reading."""
        item = _mapping(data, name="evidence", fields=cls._FIELDS)
        return cls(
            object_id=_text(item["object_id"], name="object_id"),
            name=_text(item["name"], name="name"),
            label=_text(item["label"], name="label"),
            value=_non_negative(item["value"], name="value"),
            unit=_text(item["unit"], name="unit"),
            threshold=_non_negative(item["threshold"], name="threshold"),
            max_value=_non_negative(item["max_value"], name="max_value"),
        )


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    """EgoPersist decision and the instant-update comparison."""

    object_id: str
    display_name: str
    label: LabelClass
    action: DecisionAction
    confidence: float
    reason: str
    probabilities: dict[LabelClass, float]
    instant_label: LabelClass
    instant_action: DecisionAction
    instant_confidence: float
    instant_reason: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "object_id",
            "display_name",
            "label",
            "action",
            "confidence",
            "reason",
            "probabilities",
            "instant_label",
            "instant_action",
            "instant_confidence",
            "instant_reason",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "object_id",
            "display_name",
            "reason",
            "instant_reason",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ReplaySchemaError(f"{name} must be a non-empty string")
        if not isinstance(self.label, LabelClass):
            raise ReplaySchemaError("label must be a LabelClass")
        if not isinstance(self.action, DecisionAction):
            raise ReplaySchemaError("action must be a DecisionAction")
        if not isinstance(self.instant_label, LabelClass):
            raise ReplaySchemaError("instant_label must be a LabelClass")
        if not isinstance(self.instant_action, DecisionAction):
            raise ReplaySchemaError("instant_action must be a DecisionAction")
        object.__setattr__(self, "confidence", _ratio(self.confidence, name="confidence"))
        object.__setattr__(
            self,
            "instant_confidence",
            _ratio(self.instant_confidence, name="instant_confidence"),
        )
        probabilities = {
            _label(key, name="probabilities key"): _ratio(
                value, name=f"probabilities[{key}]"
            )
            for key, value in self.probabilities.items()
        }
        if set(probabilities) != set(LabelClass):
            raise ReplaySchemaError("probabilities must contain every label")
        if abs(sum(probabilities.values()) - 1.0) > 1e-5:
            raise ReplaySchemaError("probabilities must sum to 1.0")
        object.__setattr__(self, "probabilities", probabilities)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "object_id": self.object_id,
            "display_name": self.display_name,
            "label": self.label.value,
            "action": self.action.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "probabilities": {
                label.value: value for label, value in self.probabilities.items()
            },
            "instant_label": self.instant_label.value,
            "instant_action": self.instant_action.value,
            "instant_confidence": self.instant_confidence,
            "instant_reason": self.instant_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayDecision:
        """Build and validate one decision."""
        item = _mapping(data, name="decision", fields=cls._FIELDS)
        probability_data = item["probabilities"]
        if not isinstance(probability_data, dict):
            raise ReplaySchemaError("probabilities must be an object")
        probabilities = {
            _label(key, name="probabilities key"): _ratio(
                value, name=f"probabilities[{key}]"
            )
            for key, value in probability_data.items()
        }
        return cls(
            object_id=_text(item["object_id"], name="object_id"),
            display_name=_text(item["display_name"], name="display_name"),
            label=_label(item["label"], name="label"),
            action=_action(item["action"], name="action"),
            confidence=_ratio(item["confidence"], name="confidence"),
            reason=_text(item["reason"], name="reason"),
            probabilities=probabilities,
            instant_label=_label(item["instant_label"], name="instant_label"),
            instant_action=_action(item["instant_action"], name="instant_action"),
            instant_confidence=_ratio(
                item["instant_confidence"], name="instant_confidence"
            ),
            instant_reason=_text(item["instant_reason"], name="instant_reason"),
        )


@dataclass(frozen=True, slots=True)
class ReplayQuery:
    """Answer to the mug query at one replay frame."""

    object_id: str
    object_name: str
    state: LabelClass
    position_m: Position3
    confidence: float
    answer: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "object_id",
            "object_name",
            "state",
            "position_m",
            "confidence",
            "answer",
        }
    )

    def __post_init__(self) -> None:
        for name in ("object_id", "object_name", "answer"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ReplaySchemaError(f"{name} must be a non-empty string")
        if not isinstance(self.state, LabelClass):
            raise ReplaySchemaError("state must be a LabelClass")
        object.__setattr__(
            self, "position_m", _position(self.position_m, name="position_m")
        )
        object.__setattr__(
            self, "confidence", _ratio(self.confidence, name="confidence")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "object_id": self.object_id,
            "object_name": self.object_name,
            "state": self.state.value,
            "position_m": list(self.position_m),
            "confidence": self.confidence,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayQuery:
        """Build and validate one query answer."""
        item = _mapping(data, name="query", fields=cls._FIELDS)
        return cls(
            object_id=_text(item["object_id"], name="object_id"),
            object_name=_text(item["object_name"], name="object_name"),
            state=_label(item["state"], name="state"),
            position_m=_position(item["position_m"], name="position_m"),
            confidence=_ratio(item["confidence"], name="confidence"),
            answer=_text(item["answer"], name="answer"),
        )


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """One causal keyframe in a scenario."""

    timestamp_s: float
    caption: str
    observations: tuple[ReplayObservation, ...]
    ego_map: tuple[ReplayMapEntry, ...]
    instant_map: tuple[ReplayMapEntry, ...]
    decisions: tuple[ReplayDecision, ...]
    evidence: tuple[ReplayEvidence, ...]
    query: ReplayQuery

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "timestamp_s",
            "caption",
            "observations",
            "ego_map",
            "instant_map",
            "decisions",
            "evidence",
            "query",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp_s", _non_negative(self.timestamp_s, name="timestamp_s")
        )
        if not isinstance(self.caption, str) or not self.caption:
            raise ReplaySchemaError("caption must be a non-empty string")
        for name in (
            "observations",
            "ego_map",
            "instant_map",
            "decisions",
            "evidence",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise ReplaySchemaError(f"{name} must be a tuple")
        if not all(
            isinstance(item, ReplayObservation) for item in self.observations
        ):
            raise ReplaySchemaError("observations must contain ReplayObservation")
        for name in ("ego_map", "instant_map"):
            entries = getattr(self, name)
            if not all(isinstance(item, ReplayMapEntry) for item in entries):
                raise ReplaySchemaError(f"{name} must contain ReplayMapEntry")
            ids = tuple(item.object_id for item in getattr(self, name))
            if len(set(ids)) != len(ids):
                raise ReplaySchemaError(f"{name} object IDs must be unique")
        if not all(isinstance(item, ReplayDecision) for item in self.decisions):
            raise ReplaySchemaError("decisions must contain ReplayDecision")
        if not all(isinstance(item, ReplayEvidence) for item in self.evidence):
            raise ReplaySchemaError("evidence must contain ReplayEvidence")
        if not isinstance(self.query, ReplayQuery):
            raise ReplaySchemaError("query must be a ReplayQuery")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "timestamp_s": self.timestamp_s,
            "caption": self.caption,
            "observations": [item.to_dict() for item in self.observations],
            "ego_map": [item.to_dict() for item in self.ego_map],
            "instant_map": [item.to_dict() for item in self.instant_map],
            "decisions": [item.to_dict() for item in self.decisions],
            "evidence": [item.to_dict() for item in self.evidence],
            "query": self.query.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayFrame:
        """Build and validate one frame."""
        item = _mapping(data, name="frame", fields=cls._FIELDS)
        query_data = item["query"]
        if not isinstance(query_data, dict):
            raise ReplaySchemaError("query must be an object")
        return cls(
            timestamp_s=_non_negative(item["timestamp_s"], name="timestamp_s"),
            caption=_text(item["caption"], name="caption"),
            observations=tuple(
                ReplayObservation.from_dict(_as_dict(value, name="observation"))
                for value in _sequence(item["observations"], name="observations")
            ),
            ego_map=tuple(
                ReplayMapEntry.from_dict(_as_dict(value, name="map_entry"))
                for value in _sequence(item["ego_map"], name="ego_map")
            ),
            instant_map=tuple(
                ReplayMapEntry.from_dict(_as_dict(value, name="map_entry"))
                for value in _sequence(item["instant_map"], name="instant_map")
            ),
            decisions=tuple(
                ReplayDecision.from_dict(_as_dict(value, name="decision"))
                for value in _sequence(item["decisions"], name="decisions")
            ),
            evidence=tuple(
                ReplayEvidence.from_dict(_as_dict(value, name="evidence"))
                for value in _sequence(item["evidence"], name="evidence")
            ),
            query=ReplayQuery.from_dict(query_data),
        )


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    """One labelled point on a scenario timeline."""

    timestamp_s: float
    title: str
    detail: str
    kind: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"timestamp_s", "title", "detail", "kind"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp_s", _non_negative(self.timestamp_s, name="timestamp_s")
        )
        for name in ("title", "detail", "kind"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ReplaySchemaError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "timestamp_s": self.timestamp_s,
            "title": self.title,
            "detail": self.detail,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayEvent:
        """Build and validate one timeline event."""
        item = _mapping(data, name="event", fields=cls._FIELDS)
        return cls(
            timestamp_s=_non_negative(item["timestamp_s"], name="timestamp_s"),
            title=_text(item["title"], name="title"),
            detail=_text(item["detail"], name="detail"),
            kind=_text(item["kind"], name="kind"),
        )


@dataclass(frozen=True, slots=True)
class ReplayScenario:
    """One deterministic synthetic scenario."""

    scenario_id: str
    title: str
    summary: str
    focus_object_id: str
    duration_s: float
    frames: tuple[ReplayFrame, ...]
    key_events: tuple[ReplayEvent, ...]

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "scenario_id",
            "title",
            "summary",
            "focus_object_id",
            "duration_s",
            "frames",
            "key_events",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "scenario_id",
            "title",
            "summary",
            "focus_object_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ReplaySchemaError(f"{name} must be a non-empty string")
        duration = _non_negative(self.duration_s, name="duration_s")
        if duration <= 0.0:
            raise ReplaySchemaError("duration_s must be positive")
        object.__setattr__(self, "duration_s", duration)
        if not self.frames:
            raise ReplaySchemaError("frames must not be empty")
        if not all(isinstance(frame, ReplayFrame) for frame in self.frames):
            raise ReplaySchemaError("frames must contain ReplayFrame")
        if not all(isinstance(event, ReplayEvent) for event in self.key_events):
            raise ReplaySchemaError("key_events must contain ReplayEvent")
        timestamps = tuple(frame.timestamp_s for frame in self.frames)
        if any(
            current <= previous
            for previous, current in zip(timestamps, timestamps[1:], strict=False)
        ):
            raise ReplaySchemaError("frame timestamps must increase")
        if timestamps[-1] > duration + 1e-9:
            raise ReplaySchemaError("frame timestamps must not exceed duration_s")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "summary": self.summary,
            "focus_object_id": self.focus_object_id,
            "duration_s": self.duration_s,
            "frames": [frame.to_dict() for frame in self.frames],
            "key_events": [event.to_dict() for event in self.key_events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayScenario:
        """Build and validate one scenario."""
        item = _mapping(data, name="scenario", fields=cls._FIELDS)
        return cls(
            scenario_id=_text(item["scenario_id"], name="scenario_id"),
            title=_text(item["title"], name="title"),
            summary=_text(item["summary"], name="summary"),
            focus_object_id=_text(item["focus_object_id"], name="focus_object_id"),
            duration_s=_non_negative(item["duration_s"], name="duration_s"),
            frames=tuple(
                ReplayFrame.from_dict(_as_dict(value, name="frame"))
                for value in _sequence(item["frames"], name="frames")
            ),
            key_events=tuple(
                ReplayEvent.from_dict(_as_dict(value, name="event"))
                for value in _sequence(item["key_events"], name="key_events")
            ),
        )

    def frame_at(self, timestamp_s: float) -> ReplayFrame:
        """Return the latest frame at or before the requested time."""
        timestamp = _non_negative(timestamp_s, name="timestamp_s")
        if timestamp < self.frames[0].timestamp_s:
            raise ReplaySchemaError("timestamp_s is before the first frame")
        selected = self.frames[0]
        for frame in self.frames:
            if frame.timestamp_s > timestamp:
                break
            selected = frame
        return selected


@dataclass(frozen=True, slots=True)
class ReplayDocument:
    """Top-level replay file for the static viewer and video renderer."""

    schema_version: str
    title: str
    scenarios: tuple[ReplayScenario, ...]

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema_version", "title", "scenarios"}
    )

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ReplaySchemaError(f"schema_version must be {SCHEMA_VERSION}")
        if not isinstance(self.title, str) or not self.title:
            raise ReplaySchemaError("title must be a non-empty string")
        if not self.scenarios:
            raise ReplaySchemaError("scenarios must not be empty")
        ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        if len(set(ids)) != len(ids):
            raise ReplaySchemaError("scenario_id values must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    def to_json(self) -> str:
        """Return stable JSON text."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayDocument:
        """Build and validate a replay document."""
        item = _mapping(data, name="document", fields=cls._FIELDS)
        return cls(
            schema_version=_text(item["schema_version"], name="schema_version"),
            title=_text(item["title"], name="title"),
            scenarios=tuple(
                ReplayScenario.from_dict(_as_dict(value, name="scenario"))
                for value in _sequence(item["scenarios"], name="scenarios")
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> ReplayDocument:
        """Build and validate a replay document from JSON text."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ReplaySchemaError("invalid JSON text") from exc
        if not isinstance(data, dict):
            raise ReplaySchemaError("document JSON root must be an object")
        return cls.from_dict(data)

    def scenario(self, scenario_id: str) -> ReplayScenario:
        """Return one scenario by stable identifier."""
        for item in self.scenarios:
            if item.scenario_id == scenario_id:
                return item
        raise ReplaySchemaError(f"unknown scenario: {scenario_id}")


def _as_dict(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplaySchemaError(f"{name} must be an object")
    return value
