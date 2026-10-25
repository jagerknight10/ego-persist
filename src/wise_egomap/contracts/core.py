"""Typed contracts for data exchange between WISE-EgoMap modules.

Lengths use metres. Times use seconds. A pose maps camera coordinates to world
coordinates unless a field name says otherwise. All records are immutable and
round-trip through JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum, StrEnum
from types import UnionType
from typing import Any, ClassVar, TypeVar, Union, get_args, get_origin, get_type_hints

RecordT = TypeVar("RecordT", bound="_Record")
Position3 = tuple[float, float, float]
QuaternionWXYZ = tuple[float, float, float, float]


class LabelClass(StrEnum):
    """Causal state of one 3D instance track."""

    STABLE = "STABLE"
    TRANSIENT = "TRANSIENT"
    PERSISTENT_CHANGE = "PERSISTENT_CHANGE"
    UNKNOWN = "UNKNOWN"


class EventType(StrEnum):
    """Physical event that a labelled window can contain."""

    STABLE = "stable"
    TRANSIENT = "transient"
    PERSISTENT_CHANGE = "persistent_change"
    OCCLUSION = "occlusion"
    REVERSION = "reversion"


class DecisionAction(StrEnum):
    """Map action selected by the state machine."""

    KEEP = "keep"
    COMMIT = "commit"
    DEFER = "defer"
    REJECT = "reject"
    REVERT = "revert"


def _finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _non_negative(value: float, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _unit(value: float, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _position(value: Any, name: str) -> Position3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must have three values")
    items = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    return items  # type: ignore[return-value]


def _quaternion(value: Any, name: str) -> QuaternionWXYZ:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must have four values")
    items = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    norm = math.sqrt(sum(item * item for item in items))
    if norm <= 0.0:
        raise ValueError(f"{name} must have a non-zero norm")
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"{name} must be a unit quaternion")
    return items  # type: ignore[return-value]


def _hash(value: str, name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {_encode(key): _encode(item) for key, item in value.items()}
    return value


def _decode(value: Any, hint: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(hint)
    args = get_args(hint)

    if origin in (Union, UnionType):
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _decode(value, arg)
            except (TypeError, ValueError):
                continue
        raise ValueError("value does not match any union member")

    if origin in (tuple, list):
        if not isinstance(value, (tuple, list)):
            raise ValueError("value must be a sequence")
        item_hint = args[0] if args else Any
        return [_decode(item, item_hint) for item in value]

    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError("value must be an object")
        key_hint, value_hint = args
        return {
            _decode(key, key_hint): _decode(item, value_hint)
            for key, item in value.items()
        }

    if hint is datetime:
        if not isinstance(value, str):
            raise ValueError("datetime value must be a string")
        return datetime.fromisoformat(value)

    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint(value)

    if isinstance(hint, type) and issubclass(hint, _Record):
        if not isinstance(value, dict):
            raise ValueError("record value must be an object")
        return hint.from_dict(value)

    return value


class _Record:
    """Base class with strict dictionary and JSON conversion."""

    _field_names: ClassVar[frozenset[str]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            field.name: _encode(getattr(self, field.name)) for field in fields(self)
        }

    @classmethod
    def from_dict(cls: type[RecordT], data: dict[str, Any]) -> RecordT:
        """Build a record and reject unknown or missing fields."""
        names = {field.name for field in fields(cls)}
        unknown = set(data) - names
        missing = names - set(data)
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing fields: {sorted(missing)}")
        hints = get_type_hints(cls)
        values = {name: _decode(data[name], hints[name]) for name in names}
        return cls(**values)

    def to_json(self) -> str:
        """Return stable JSON text."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls: type[RecordT], text: str) -> RecordT:
        """Build a record from JSON text."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class Observation(_Record):
    """One mask observation of an instance at a causal keyframe."""

    sequence_id: str
    frame_id: str
    timestamp_s: float
    instance_id: str
    mask_area_px: int
    depth_support: float
    co_visibility: float
    appearance_similarity: float
    centroid_world_m: Position3
    extent_world_m: Position3
    orientation_wxyz: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)
    displacement_m: float = 0.0
    rotation_deg: float = 0.0
    valid_depth_points: int = 0
    position_covariance_m2: tuple[float, ...] = (0.0,) * 9
    geometry_source: str = "legacy_gt"

    def __post_init__(self) -> None:
        for name in ("sequence_id", "frame_id", "instance_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self, "timestamp_s", _non_negative(self.timestamp_s, "timestamp_s")
        )
        if not isinstance(self.mask_area_px, int) or isinstance(
            self.mask_area_px, bool
        ):
            raise TypeError("mask_area_px must be an integer")
        if self.mask_area_px <= 0:
            raise ValueError("mask_area_px must be positive")
        object.__setattr__(
            self, "depth_support", _unit(self.depth_support, "depth_support")
        )
        object.__setattr__(
            self, "co_visibility", _unit(self.co_visibility, "co_visibility")
        )
        object.__setattr__(
            self,
            "appearance_similarity",
            _unit(self.appearance_similarity, "appearance_similarity"),
        )
        object.__setattr__(
            self,
            "centroid_world_m",
            _position(self.centroid_world_m, "centroid_world_m"),
        )
        extent = _position(self.extent_world_m, "extent_world_m")
        if any(item <= 0.0 for item in extent):
            raise ValueError("extent_world_m values must be positive")
        object.__setattr__(self, "extent_world_m", extent)
        object.__setattr__(
            self,
            "orientation_wxyz",
            _quaternion(self.orientation_wxyz, "orientation_wxyz"),
        )
        object.__setattr__(
            self, "displacement_m", _non_negative(self.displacement_m, "displacement_m")
        )
        object.__setattr__(
            self, "rotation_deg", _non_negative(self.rotation_deg, "rotation_deg")
        )
        if (
            not isinstance(self.valid_depth_points, int)
            or isinstance(self.valid_depth_points, bool)
            or self.valid_depth_points < 0
        ):
            raise ValueError("valid_depth_points must be a non-negative integer")
        covariance = tuple(
            _finite(item, f"position_covariance_m2[{index}]")
            for index, item in enumerate(self.position_covariance_m2)
        )
        if len(covariance) != 9:
            raise ValueError("position_covariance_m2 must contain nine values")
        if any(item < 0.0 for item in (covariance[0], covariance[4], covariance[8])):
            raise ValueError("position covariance diagonal must be non-negative")
        object.__setattr__(self, "position_covariance_m2", covariance)
        if self.geometry_source not in {"rgbd", "legacy_gt"}:
            raise ValueError("geometry_source must be 'rgbd' or 'legacy_gt'")


@dataclass(frozen=True, slots=True)
class EvidenceVector(_Record):
    """Named evidence values for one track observation."""

    feature_names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        values = tuple(
            _finite(item, f"values[{index}]") for index, item in enumerate(self.values)
        )
        if not names:
            raise ValueError("feature_names must not be empty")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        if len(names) != len(values):
            raise ValueError("feature_names and values must have equal length")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class Decision(_Record):
    """Causal decision for one track at one keyframe."""

    sequence_id: str
    track_id: str
    frame_id: str
    timestamp_s: float
    label: LabelClass
    action: DecisionAction
    confidence: float
    probabilities: dict[LabelClass, float]
    reason: str

    def __post_init__(self) -> None:
        for name in ("sequence_id", "track_id", "frame_id", "reason"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self, "timestamp_s", _non_negative(self.timestamp_s, "timestamp_s")
        )
        if not isinstance(self.label, LabelClass):
            raise TypeError("label must be a LabelClass")
        if not isinstance(self.action, DecisionAction):
            raise TypeError("action must be a DecisionAction")
        object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))
        probabilities = {
            LabelClass(key): _unit(value, f"probabilities[{key}]")
            for key, value in self.probabilities.items()
        }
        if set(probabilities) != set(LabelClass):
            raise ValueError("probabilities must contain every label exactly once")
        if abs(sum(probabilities.values()) - 1.0) > 1e-5:
            raise ValueError("probabilities must sum to 1.0")
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True, slots=True)
class MapUpdate(_Record):
    """Map transaction selected by a decision."""

    sequence_id: str
    track_id: str
    timestamp_s: float
    action: DecisionAction
    map_version: int
    previous_position_m: Position3 | None = None
    new_position_m: Position3 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("sequence_id must be a non-empty string")
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a non-empty string")
        object.__setattr__(
            self, "timestamp_s", _non_negative(self.timestamp_s, "timestamp_s")
        )
        if not isinstance(self.action, DecisionAction):
            raise TypeError("action must be a DecisionAction")
        if not isinstance(self.map_version, int) or isinstance(self.map_version, bool):
            raise TypeError("map_version must be an integer")
        if self.map_version < 0:
            raise ValueError("map_version must be non-negative")
        if self.previous_position_m is not None:
            object.__setattr__(
                self,
                "previous_position_m",
                _position(self.previous_position_m, "previous_position_m"),
            )
        if self.new_position_m is not None:
            object.__setattr__(
                self, "new_position_m", _position(self.new_position_m, "new_position_m")
            )
        needs_new = self.action in (DecisionAction.COMMIT, DecisionAction.REVERT)
        if needs_new and self.new_position_m is None:
            raise ValueError(f"{self.action.value} requires new_position_m")


@dataclass(frozen=True, slots=True)
class EventLabel(_Record):
    """Ground-truth event window from label rules or manual audit."""

    sequence_id: str
    track_id: str
    event_type: EventType
    start_s: float
    end_s: float
    label: LabelClass
    source: str = "rules_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("sequence_id must be a non-empty string")
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a non-empty string")
        if not isinstance(self.event_type, EventType):
            raise TypeError("event_type must be an EventType")
        if not isinstance(self.label, LabelClass):
            raise TypeError("label must be a LabelClass")
        object.__setattr__(self, "start_s", _non_negative(self.start_s, "start_s"))
        object.__setattr__(self, "end_s", _non_negative(self.end_s, "end_s"))
        if self.end_s < self.start_s:
            raise ValueError("end_s must be greater than or equal to start_s")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")


@dataclass(frozen=True, slots=True)
class TrackObservation(_Record):
    """Observation joined to its assigned persistent track."""

    track_id: str
    observation: Observation
    committed_position_m: Position3
    committed_orientation_wxyz: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a non-empty string")
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation")
        object.__setattr__(
            self,
            "committed_position_m",
            _position(self.committed_position_m, "committed_position_m"),
        )
        object.__setattr__(
            self,
            "committed_orientation_wxyz",
            _quaternion(self.committed_orientation_wxyz, "committed_orientation_wxyz"),
        )


@dataclass(frozen=True, slots=True)
class RunManifest(_Record):
    """Reproducibility record written by every run."""

    run_id: str
    git_commit: str
    git_dirty: bool
    command: tuple[str, ...]
    config_path: str
    config_hash: str
    data_split_hash: str
    label_config_hash: str
    seed: int
    started_at_utc: datetime
    hostname: str
    python_version: str
    packages: dict[str, str]
    extra: dict[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "git_commit",
            "config_path",
            "hostname",
            "python_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if len(self.git_commit) < 7:
            raise ValueError("git_commit must have at least seven characters")
        for name in ("config_hash", "data_split_hash", "label_config_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), name))
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.started_at_utc, datetime):
            raise TypeError("started_at_utc must be a datetime")
        if self.started_at_utc.tzinfo is None:
            raise ValueError("started_at_utc must include a timezone")
        command = tuple(self.command)
        if not command or any(not isinstance(item, str) for item in command):
            raise ValueError("command must contain strings")
        object.__setattr__(self, "command", command)
        object.__setattr__(
            self,
            "packages",
            {str(key): str(value) for key, value in self.packages.items()},
        )
        object.__setattr__(self, "extra", dict(self.extra))
