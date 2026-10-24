"""Gated nearest-association of observations to tracks.

Matches one keyframe's observations to existing tracks with the
Hungarian algorithm. The cost combines centroid distance (normalised
by the gate) and 1 - IoU of the oriented boxes. Pairs above the gate
are not allowed. Unmatched observations start provisional tracks; a
provisional track is confirmed after ``confirm_hits`` matches.

Only causal information is used: the matcher sees the committed track
poses and the current keyframe's observations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from egopersist.contracts.core import Observation
from egopersist.geometry.obb import OBB, centroid_distance_m, iou_m3

_SENTINEL = 1e6


def default_gate(expected_motion_m: float) -> float:
    """Association gate: at least 0.3 m, or twice the expected motion."""
    return max(0.3, 2.0 * float(expected_motion_m))


@dataclass
class Track:
    """Mutable causal track state."""

    track_id: str
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    extent_m: tuple[float, float, float]
    first_s: float
    last_s: float
    hits: int
    confirmed: bool = False


@dataclass
class AssociationResult:
    """Matches and unmatched observation indices for one keyframe."""

    matches: list[tuple[int, int]] = field(default_factory=list)
    unmatched_obs: list[int] = field(default_factory=list)

    def matched_obs(self) -> set[int]:
        return {obs_idx for _, obs_idx in self.matches}


def _obb(obs: Observation) -> OBB:
    return OBB(
        centroid_m=obs.centroid_world_m,
        half_extents_m=tuple(v / 2.0 for v in obs.extent_world_m),  # type: ignore[call-arg]
        orientation_wxyz=obs.orientation_wxyz,
    )


def compute_cost_matrix(
    tracks: Sequence[Track],
    observations: Sequence[Observation],
    gate_m: float,
) -> np.ndarray:
    """Cost per (track, observation); gated pairs get the sentinel."""
    cost = np.full((len(tracks), len(observations)), _SENTINEL, dtype=float)
    for i, track in enumerate(tracks):
        track_box = OBB(
            centroid_m=track.position_m,
            half_extents_m=tuple(v / 2.0 for v in track.extent_m),  # type: ignore[call-arg]
            orientation_wxyz=track.orientation_wxyz,
        )
        for j, obs in enumerate(observations):
            dist = centroid_distance_m(track_box, _obb(obs))
            if dist > gate_m:
                continue
            iou = iou_m3(track_box, _obb(obs))
            cost[i, j] = dist / gate_m + (1.0 - iou)
    return cost


def match(cost: np.ndarray, gate_m: float) -> list[tuple[int, int]]:
    """Hungarian matching; drop pairs that exceed the gate."""
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return []
    row, col = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row, col):
        if cost[i, j] >= _SENTINEL:
            continue
        if cost[i, j] > 2.0:  # dist/gate + (1 - IoU) max ~2 when in gate
            continue
        matches.append((int(i), int(j)))
    return matches


class TrackBank:
    """Causal track memory for one sequence."""

    def __init__(self, confirm_hits: int = 3, id_prefix: str = "trk"):
        if confirm_hits < 1:
            raise ValueError("confirm_hits must be >= 1")
        self.confirm_hits = confirm_hits
        self.tracks: list[Track] = []
        self._next_id = 0
        self._prefix = id_prefix

    def _make_id(self) -> str:
        self._next_id += 1
        return f"{self._prefix}{self._next_id:04d}"

    def add_provisional(self, obs: Observation) -> Track:
        track = Track(
            track_id=self._make_id(),
            position_m=obs.centroid_world_m,
            orientation_wxyz=obs.orientation_wxyz,
            extent_m=obs.extent_world_m,
            first_s=obs.timestamp_s,
            last_s=obs.timestamp_s,
            hits=1,
            confirmed=self.confirm_hits <= 1,
        )
        self.tracks.append(track)
        return track

    def associate(self, observations: Sequence[Observation], gate_m: float) -> AssociationResult:
        result = AssociationResult()
        if not observations:
            return result
        if not self.tracks:
            result.unmatched_obs = list(range(len(observations)))
            return result
        cost = compute_cost_matrix(self.tracks, observations, gate_m)
        result.matches = match(cost, gate_m)
        taken = result.matched_obs()
        result.unmatched_obs = [j for j in range(len(observations)) if j not in taken]
        return result

    def apply(self, result: AssociationResult, observations: Sequence[Observation]) -> list[Track]:
        """Update track states in place; return new provisional tracks."""
        for i, j in result.matches:
            obs = observations[j]
            track = self.tracks[i]
            track.position_m = obs.centroid_world_m
            track.orientation_wxyz = obs.orientation_wxyz
            track.extent_m = obs.extent_world_m
            track.last_s = obs.timestamp_s
            track.hits += 1
            track.confirmed = track.hits >= self.confirm_hits
        new_tracks = [self.add_provisional(observations[j]) for j in result.unmatched_obs]
        return new_tracks

    def missing_tracks(self, now_s: float) -> list[Track]:
        """Tracks with no observation at the current keyframe."""
        return [t for t in self.tracks if t.last_s < now_s - 1e-9]
