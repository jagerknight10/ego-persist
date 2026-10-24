"""Causal per-track state machine for map maintenance.

One instance per sequence; ``update()`` is called per track per 2 Hz
keyframe with the causal track view. Each update returns a
``Decision`` (state + action + confidence + reason) and a
``MapUpdate`` (the map transaction).

Rules (thresholds from ``configs/labels_v1.yaml``):

* Static tracks keep their map position while observations agree
  (deviation <= change threshold).
* A deviation beyond the change threshold opens a suspect window.
  The map is untouched while the window is open (action defer).
* Commit requires all gates: P(persistent_change) >= 0.85, >= 5
  visible keyframes in the window, span >= 1.0 s, mean co-visibility
  >= 0.60, and a stable new position (mode within 0.05 m). Commit
  moves the map entry and increments the map version.
* A committed change that returns to its pre-change pose within
  0.05 m / 10 deg reverts the map entry (version +1).
* Dynamic tracks become TRANSIENT once evidence is sufficient. A
  transient element that stays out of view beyond the 15 s return
  window is rejected: it is not a persistent map element.
* Non-observation never removes a persistent map element. Any
  occluded static track defers; its map entry survives.

Only causal information is used: the update at time t sees
observations at or before t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from egopersist.contracts.core import (
    Decision,
    DecisionAction,
    LabelClass,
    MapUpdate,
)
from egopersist.geometry.obb import OBB, centroid_distance_m
from egopersist.geometry.transforms import rotation_angle_deg

Position3 = tuple[float, float, float]
QuaternionWXYZ = tuple[float, float, float, float]


@dataclass(frozen=True)
class StateMachineConfig:
    """Thresholds from configs/labels_v1.yaml."""

    change_translation_m: float = 0.10
    change_rotation_deg: float = 15.0
    revert_tolerance_m: float = 0.05
    revert_tolerance_deg: float = 10.0
    persistent_mode_tolerance_m: float = 0.05
    transient_return_window_s: float = 15.0
    suspect_probability: float = 0.50
    commit_probability: float = 0.85
    commit_min_visible_keyframes: int = 5
    commit_min_span_s: float = 1.0
    commit_min_co_visibility: float = 0.60
    transient_probability: float = 0.80

    @classmethod
    def from_labels_yaml(cls, raw: Mapping[str, object]) -> "StateMachineConfig":
        change = raw.get("change", {}) or {}
        persistent = raw.get("persistent", {}) or {}
        transient = raw.get("transient", {}) or {}
        sm = raw.get("state_machine", {}) or {}
        return cls(
            change_translation_m=float(change["translation_m"]),  # type: ignore[arg-type]
            change_rotation_deg=float(change["rotation_deg"]),  # type: ignore[arg-type]
            revert_tolerance_m=float(persistent["mode_tolerance_m"]),  # type: ignore[arg-type]
            revert_tolerance_deg=float(persistent["mode_tolerance_deg"]),  # type: ignore[arg-type]
            persistent_mode_tolerance_m=float(  # type: ignore[arg-type]
                persistent["mode_tolerance_m"]
            ),
            transient_return_window_s=float(transient["return_window_s"]),  # type: ignore[arg-type]
            suspect_probability=float(sm["suspect_probability"]),  # type: ignore[arg-type]
            commit_probability=float(sm["commit_probability"]),  # type: ignore[arg-type]
            commit_min_visible_keyframes=int(  # type: ignore[arg-type]
                sm["commit_min_visible_keyframes"]
            ),
            commit_min_span_s=float(sm["commit_min_span_s"]),  # type: ignore[arg-type]
            commit_min_co_visibility=float(sm["commit_min_co_visibility"]),  # type: ignore[arg-type]
            transient_probability=float(sm["transient_probability"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class TrackView:
    """Everything the state machine may see at one keyframe (causal).

    ``observation`` is ``None`` when the track is occluded at this
    keyframe. ``probabilities`` are the causal evidence model's class
    probabilities for this track at this keyframe.
    """

    sequence_id: str
    track_id: str
    frame_id: str
    timestamp_s: float
    observation: object | None  # Observation | None (kept loose for callers)
    probabilities: Mapping[LabelClass, float]
    committed_position_m: Position3
    committed_orientation_wxyz: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)
    is_static: bool = True


@dataclass
class _Memory:
    label: LabelClass = LabelClass.UNKNOWN
    committed: Position3 | None = None
    committed_q: QuaternionWXYZ | None = None
    original: Position3 | None = None
    original_q: QuaternionWXYZ | None = None
    pre_change: Position3 | None = None
    pre_change_q: QuaternionWXYZ | None = None
    suspect_start_s: float | None = None
    suspect_positions: list[Position3] = field(default_factory=list)
    suspect_co_vis: list[float] = field(default_factory=list)
    visible_keyframes: int = 0
    last_seen_s: float | None = None
    first_seen_s: float | None = None
    map_version: int = 0


def _dist(a: Position3, b: Position3) -> float:
    return math.dist(a, b)


def _median(points: list[Position3]) -> Position3:
    import statistics

    return tuple(  # type: ignore[return-value]
        statistics.median(p[axis] for p in points) for axis in range(3)
    )


def _normalize_probabilities(
    probabilities: Mapping[LabelClass, float],
) -> dict[LabelClass, float]:
    values = {label: max(0.0, float(probabilities.get(label, 0.0))) for label in LabelClass}
    total = sum(values.values())
    if total <= 0.0:
        even = 1.0 / len(LabelClass)
        return {label: even for label in LabelClass}
    return {label: value / total for label, value in values.items()}


def _deviation(
    obs: object,
    committed: Position3,
    committed_q: QuaternionWXYZ,
) -> tuple[float, float]:
    """(translation, rotation) deviation of an observation from the map."""
    extent = tuple(v / 2.0 for v in obs.extent_world_m)  # type: ignore[attr-defined]
    box = OBB(
        centroid_m=obs.centroid_world_m,  # type: ignore[attr-defined]
        half_extents_m=extent,
        orientation_wxyz=obs.orientation_wxyz,  # type: ignore[attr-defined]
    )
    committed_box = OBB(
        centroid_m=committed,
        half_extents_m=extent,
        orientation_wxyz=committed_q,
    )
    return (
        centroid_distance_m(box, committed_box),
        rotation_angle_deg(
            obs.orientation_wxyz,  # type: ignore[attr-defined]
            committed_q,
        ),
    )


class TrackStateMachine:
    """Causal decision maker for one sequence's tracks."""

    def __init__(self, config: StateMachineConfig):
        self.config = config
        self.memory: dict[str, _Memory] = {}

    # -- helpers ----------------------------------------------------------

    def _get(self, view: TrackView) -> _Memory:
        mem = self.memory.get(view.track_id)
        if mem is None:
            mem = _Memory(
                committed=view.committed_position_m,
                committed_q=view.committed_orientation_wxyz,
                original=view.committed_position_m,
                original_q=view.committed_orientation_wxyz,
            )
            self.memory[view.track_id] = mem
        return mem

    def _decision(
        self,
        view: TrackView,
        label: LabelClass,
        action: DecisionAction,
        reason: str,
        probs: dict[LabelClass, float],
    ) -> Decision:
        return Decision(
            sequence_id=view.sequence_id,
            track_id=view.track_id,
            frame_id=view.frame_id,
            timestamp_s=view.timestamp_s,
            label=label,
            action=action,
            confidence=probs[label],
            probabilities=probs,
            reason=reason,
        )

    def _record(
        self,
        view: TrackView,
        mem: _Memory,
        action: DecisionAction,
        new_position: Position3 | None,
        previous_position: Position3 | None,
    ) -> MapUpdate:
        if action in (DecisionAction.COMMIT, DecisionAction.REVERT):
            mem.map_version += 1
            assert new_position is not None
            mem.committed = new_position
        return MapUpdate(
            sequence_id=view.sequence_id,
            track_id=view.track_id,
            timestamp_s=view.timestamp_s,
            action=action,
            map_version=mem.map_version,
            previous_position_m=previous_position,
            new_position_m=new_position,
        )

    def _clear_suspect(self, mem: _Memory) -> None:
        mem.suspect_start_s = None
        mem.suspect_positions = []
        mem.suspect_co_vis = []

    # -- update -----------------------------------------------------------

    def update(self, view: TrackView) -> tuple[Decision, MapUpdate]:
        cfg = self.config
        mem = self._get(view)
        probs = _normalize_probabilities(view.probabilities)
        committed = mem.committed or view.committed_position_m
        committed_q = mem.committed_q or view.committed_orientation_wxyz

        obs = view.observation
        if obs is None:
            return self._on_missing(view, mem, probs)

        if mem.first_seen_s is None:
            mem.first_seen_s = view.timestamp_s
        mem.visible_keyframes += 1
        mem.last_seen_s = view.timestamp_s

        if not view.is_static:
            return self._on_dynamic(view, mem, probs)

        dev_m, dev_deg = _deviation(obs, committed, committed_q)
        exceeds = dev_m > cfg.change_translation_m or dev_deg > cfg.change_rotation_deg

        if not exceeds:
            return self._on_agree(view, mem, probs, committed_q)

        # --- deviation from the map --------------------------------------
        # Revert gate: committed change, object back at pre-change pose.
        if (
            mem.label == LabelClass.PERSISTENT_CHANGE
            and mem.pre_change is not None
            and mem.pre_change_q is not None
            and mem.original is not None
        ):
            back_m = _dist(obs.centroid_world_m, mem.pre_change)  # type: ignore[attr-defined]
            back_deg = rotation_angle_deg(
                obs.orientation_wxyz,  # type: ignore[attr-defined]
                mem.pre_change_q,
            )
            if back_m <= cfg.revert_tolerance_m and back_deg <= cfg.revert_tolerance_deg:
                prev = committed
                mem.label = LabelClass.STABLE
                mem.committed_q = view.committed_orientation_wxyz
                mem.pre_change = None
                mem.pre_change_q = None
                self._clear_suspect(mem)
                reason = (
                    f"revert: returned to pre-change pose "
                    f"({back_m:.3f} m, {back_deg:.1f} deg within tolerance)"
                )
                update = self._record(
                    view, mem, DecisionAction.REVERT, mem.original, prev
                )
                return (
                    self._decision(view, mem.label, DecisionAction.REVERT, reason, probs),
                    update,
                )

        if mem.suspect_start_s is None:
            mem.suspect_start_s = view.timestamp_s
            mem.suspect_positions = [obs.centroid_world_m]  # type: ignore[attr-defined]
            mem.suspect_co_vis = [obs.co_visibility]  # type: ignore[attr-defined]
        else:
            mem.suspect_positions.append(obs.centroid_world_m)  # type: ignore[attr-defined]
            mem.suspect_co_vis.append(obs.co_visibility)  # type: ignore[attr-defined]

        p_change = probs[LabelClass.PERSISTENT_CHANGE]
        if p_change < cfg.suspect_probability:
            # likely sensor noise; map untouched, no suspect record
            self._clear_suspect(mem)
            action = (
                DecisionAction.KEEP
                if mem.label != LabelClass.UNKNOWN
                else DecisionAction.DEFER
            )
            reason = (
                f"keep: deviation {dev_m:.3f} m not persistent "
                f"(P={p_change:.2f} < suspect {cfg.suspect_probability:.2f})"
            )
            update = self._record(view, mem, action, None, None)
            return self._decision(view, mem.label, action, reason, probs), update

        # genuine suspect: evaluate the commit gates
        window = mem.suspect_positions
        mode = _median(window)
        start = mem.suspect_start_s
        start = start if start is not None else view.timestamp_s
        span = view.timestamp_s - start
        n_vis = len(window)
        mean_co_vis = (
            sum(mem.suspect_co_vis) / len(mem.suspect_co_vis)
            if mem.suspect_co_vis
            else 0.0
        )
        stable_new = all(_dist(p, mode) <= cfg.persistent_mode_tolerance_m for p in window)
        gates = {
            "probability": p_change >= cfg.commit_probability,
            "visible_keyframes": n_vis >= cfg.commit_min_visible_keyframes,
            "span": span >= cfg.commit_min_span_s,
            "co_visibility": mean_co_vis >= cfg.commit_min_co_visibility,
            "stable_position": stable_new,
        }
        if all(gates.values()):
            prev = committed
            mem.committed_q = view.committed_orientation_wxyz
            mem.pre_change = committed
            mem.pre_change_q = committed_q
            mem.label = LabelClass.PERSISTENT_CHANGE
            self._clear_suspect(mem)
            reason = (
                f"commit: moved {dev_m:.3f} m, held {span:.1f} s over "
                f"{n_vis} keyframes, P={p_change:.2f}, co-vis {mean_co_vis:.2f}"
            )
            update = self._record(view, mem, DecisionAction.COMMIT, mode, prev)
            return (
                self._decision(
                    view, LabelClass.PERSISTENT_CHANGE, DecisionAction.COMMIT, reason, probs
                ),
                update,
            )

        missing = [name for name, ok in gates.items() if not ok]
        reason = (
            f"defer: change suspected since t={start:.1f} s "
            f"(dev {dev_m:.3f} m, P={p_change:.2f}); awaiting "
            f"{', '.join(missing) if missing else 'persistence'}"
        )
        update = self._record(view, mem, DecisionAction.DEFER, None, None)
        return (
            self._decision(view, LabelClass.UNKNOWN, DecisionAction.DEFER, reason, probs),
            update,
        )

    # -- branches ----------------------------------------------------------

    def _on_agree(
        self,
        view: TrackView,
        mem: _Memory,
        probs: dict[LabelClass, float],
        committed_q: QuaternionWXYZ,
    ) -> tuple[Decision, MapUpdate]:
        cfg = self.config
        obs = view.observation
        assert obs is not None
        if mem.label == LabelClass.PERSISTENT_CHANGE and mem.original is not None:
            back_m = _dist(obs.centroid_world_m, mem.original)  # type: ignore[attr-defined]
            if (
                back_m <= cfg.revert_tolerance_m
                and mem.original_q is not None
                and rotation_angle_deg(
                    obs.orientation_wxyz, mem.original_q  # type: ignore[attr-defined]
                )
                <= cfg.revert_tolerance_deg
            ):
                prev = mem.committed
                mem.label = LabelClass.STABLE
                mem.committed_q = committed_q
                mem.pre_change = None
                mem.pre_change_q = None
                self._clear_suspect(mem)
                reason = f"revert: back at original pose ({back_m:.3f} m)"
                update = self._record(view, mem, DecisionAction.REVERT, mem.original, prev)
                return (
                    self._decision(view, mem.label, DecisionAction.REVERT, reason, probs),
                    update,
                )
        self._clear_suspect(mem)
        if mem.label == LabelClass.UNKNOWN and mem.visible_keyframes >= cfg.commit_min_visible_keyframes:
            mem.label = LabelClass.STABLE
        if mem.label == LabelClass.PERSISTENT_CHANGE:
            reason = "keep: stable at changed position"
            action = DecisionAction.KEEP
        elif mem.label == LabelClass.STABLE:
            reason = "keep: position matches map"
            action = DecisionAction.KEEP
        else:
            reason = f"defer: insufficient evidence ({mem.visible_keyframes}/{cfg.commit_min_visible_keyframes} keyframes)"
            action = DecisionAction.DEFER
        update = self._record(view, mem, action, None, None)
        return self._decision(view, mem.label, action, reason, probs), update

    def _on_missing(
        self,
        view: TrackView,
        mem: _Memory,
        probs: dict[LabelClass, float],
    ) -> tuple[Decision, MapUpdate]:
        cfg = self.config
        # Never remove a persistent map element on non-observation.
        # A transient element that misses the return window is rejected.
        if (
            not view.is_static
            and mem.first_seen_s is not None
            and mem.last_seen_s is not None
            and (view.timestamp_s - mem.last_seen_s) > cfg.transient_return_window_s
        ):
            reason = (
                f"reject: transient element left at "
                f"t={mem.last_seen_s:.1f} s (window "
                f"{cfg.transient_return_window_s:.0f} s)"
            )
            update = self._record(view, mem, DecisionAction.REJECT, None, None)
            return (
                self._decision(
                    view, LabelClass.TRANSIENT, DecisionAction.REJECT, reason, probs
                ),
                update,
            )
        if mem.label == LabelClass.TRANSIENT:
            if mem.last_seen_s is not None and (
                view.timestamp_s - mem.last_seen_s
            ) > cfg.transient_return_window_s:
                label = LabelClass.TRANSIENT
                action = DecisionAction.REJECT
                reason = (
                    f"reject: transient element left at "
                    f"t={mem.last_seen_s:.1f} s (window "
                    f"{cfg.transient_return_window_s:.0f} s)"
                )
            else:
                label = LabelClass.TRANSIENT
                action = DecisionAction.KEEP
                reason = "keep: transient element briefly out of view"
        elif mem.suspect_start_s is not None:
            label = LabelClass.UNKNOWN
            action = DecisionAction.DEFER
            reason = "defer: occluded while change suspected; map kept"
        else:
            label = mem.label
            action = DecisionAction.DEFER
            reason = "defer: occluded; map element retained"
        update = self._record(view, mem, action, None, None)
        return self._decision(view, label, action, reason, probs), update

    def _on_dynamic(
        self,
        view: TrackView,
        mem: _Memory,
        probs: dict[LabelClass, float],
    ) -> tuple[Decision, MapUpdate]:
        cfg = self.config
        p_transient = probs[LabelClass.TRANSIENT]
        if (
            p_transient >= cfg.transient_probability
            and mem.visible_keyframes >= cfg.commit_min_visible_keyframes
        ):
            mem.label = LabelClass.TRANSIENT
            action = DecisionAction.KEEP
            reason = f"keep: transient object (P={p_transient:.2f})"
        else:
            action = DecisionAction.DEFER
            reason = (
                f"defer: dynamic object, evidence insufficient "
                f"(P={p_transient:.2f}, "
                f"{mem.visible_keyframes}/{cfg.commit_min_visible_keyframes} keyframes)"
            )
        update = self._record(view, mem, action, None, None)
        return (
            self._decision(view, mem.label, action, reason, probs),
            update,
        )
