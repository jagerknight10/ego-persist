"""Persistent object-centric 3D map storage."""

from __future__ import annotations

from dataclasses import dataclass, field

from wise_egomap.contracts import LabelClass, Observation


@dataclass(slots=True)
class MapEntry:
    """One persistent object entry with causal geometry history."""

    track_id: str
    category: str
    position_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    covariance_m2: tuple[float, ...]
    observation_count: int = 0
    last_seen_s: float | None = None
    state: LabelClass = LabelClass.STABLE
    confidence: float = 1.0
    map_version: int = 0
    history: list[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, ...],
        ]
    ] = field(default_factory=list)

    def observe(self, observation: Observation, timestamp_s: float) -> None:
        """Update evidence without changing persistent geometry."""
        self.observation_count += 1
        self.last_seen_s = timestamp_s
        self.confidence = max(0.0, min(1.0, observation.depth_support))

    def commit(self, observation: Observation, timestamp_s: float) -> None:
        """Commit a new persistent geometry version."""
        self.history.append(
            (self.map_version, self.position_m, self.extent_m, self.covariance_m2)
        )
        self.map_version += 1
        self.position_m = observation.centroid_world_m
        self.extent_m = observation.extent_world_m
        self.covariance_m2 = observation.position_covariance_m2
        self.observation_count += 1
        self.last_seen_s = timestamp_s
        self.state = LabelClass.PERSISTENT_CHANGE
        self.confidence = max(0.0, min(1.0, observation.depth_support))

    def revert(self, timestamp_s: float) -> bool:
        """Restore the previous persistent position, if one exists."""
        if not self.history:
            return False
        (
            _version,
            previous_position,
            previous_extent,
            previous_covariance,
        ) = self.history.pop()
        self.position_m = previous_position
        self.extent_m = previous_extent
        self.covariance_m2 = previous_covariance
        self.map_version += 1
        self.last_seen_s = timestamp_s
        return True


class PersistentMap:
    """Causal map store keyed by persistent track ID."""

    def __init__(self) -> None:
        self.entries: dict[str, MapEntry] = {}

    def add(self, entry: MapEntry) -> None:
        if entry.track_id in self.entries:
            raise ValueError(f"track already exists: {entry.track_id}")
        self.entries[entry.track_id] = entry

    def get(self, track_id: str) -> MapEntry:
        try:
            return self.entries[track_id]
        except KeyError as exc:
            raise KeyError(f"unknown map track: {track_id}") from exc
