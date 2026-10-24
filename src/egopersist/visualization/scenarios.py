"""Build the three deterministic synthetic demo scenarios."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from egopersist.contracts import LabelClass
from egopersist.visualization.policy import (
    DemoStateMachineAdapter,
    InstantUpdateAdapter,
    StateMachineAdapter,
)
from egopersist.visualization.query import answer_object_query
from egopersist.visualization.replay import (
    SCHEMA_VERSION,
    Position3,
    ReplayDecision,
    ReplayDocument,
    ReplayEvent,
    ReplayFrame,
    ReplayMapEntry,
    ReplayObservation,
    ReplayScenario,
)

KEYFRAME_S = 0.5


def _map_entry(
    object_id: str,
    display_name: str,
    kind: str,
    position_m: Position3,
    *,
    map_version: int = 0,
) -> ReplayMapEntry:
    return ReplayMapEntry(
        object_id=object_id,
        display_name=display_name,
        kind=kind,
        position_m=position_m,
        state=LabelClass.STABLE,
        confidence=0.95,
        map_version=map_version,
    )


def _observation(
    object_id: str,
    display_name: str,
    kind: str,
    position_m: Position3,
    extent_m: Position3,
    *,
    confidence: float = 0.94,
    human_caused: bool = False,
) -> ReplayObservation:
    return ReplayObservation(
        object_id=object_id,
        display_name=display_name,
        kind=kind,
        position_m=position_m,
        extent_m=extent_m,
        confidence=confidence,
        human_caused=human_caused,
    )


def _build_frame(
    *,
    sequence_id: str,
    frame_id: str,
    timestamp_s: float,
    caption: str,
    observations: Sequence[ReplayObservation],
    ego: StateMachineAdapter,
    instant: StateMachineAdapter,
    object_names: dict[str, str],
) -> ReplayFrame:
    ego_snapshot = ego.update(frame_id, timestamp_s, observations)
    instant_snapshot = instant.update(frame_id, timestamp_s, observations)
    instant_by_id = {item.track_id: item for item in instant_snapshot.decisions}

    decisions: list[ReplayDecision] = []
    for item in ego_snapshot.decisions:
        other = instant_by_id.get(item.track_id)
        decisions.append(
            ReplayDecision(
                object_id=item.track_id,
                display_name=object_names.get(item.track_id, item.track_id),
                label=item.label,
                action=item.action,
                confidence=item.confidence,
                reason=item.reason,
                probabilities=item.probabilities,
                instant_label=(
                    other.label if other is not None else LabelClass.UNKNOWN
                ),
                instant_action=(other.action if other is not None else item.action),
                instant_confidence=(other.confidence if other is not None else 0.0),
                instant_reason=(
                    other.reason
                    if other is not None
                    else "Instant map has no decision for this track."
                ),
            )
        )

    query = answer_object_query(
        "mug",
        ego_snapshot.map_entries,
        timestamp_s=timestamp_s,
    )
    return ReplayFrame(
        timestamp_s=timestamp_s,
        caption=caption,
        observations=tuple(observations),
        ego_map=ego_snapshot.map_entries,
        instant_map=instant_snapshot.map_entries,
        decisions=tuple(decisions),
        evidence=ego_snapshot.evidence,
        query=query,
    )


def _run_scenario(
    *,
    scenario_id: str,
    title: str,
    summary: str,
    focus_object_id: str,
    duration_s: float,
    initial_map: Sequence[ReplayMapEntry],
    frames: Iterable[tuple[float, str, Sequence[ReplayObservation]]],
    key_events: Sequence[ReplayEvent],
    object_names: dict[str, str],
) -> ReplayScenario:
    ego = DemoStateMachineAdapter()
    instant = InstantUpdateAdapter()
    ego.reset(scenario_id, initial_map)
    instant.reset(scenario_id, initial_map)
    replay_frames = tuple(
        _build_frame(
            sequence_id=scenario_id,
            frame_id=f"{scenario_id}-f{index:03d}",
            timestamp_s=timestamp_s,
            caption=caption,
            observations=tuple(observations),
            ego=ego,
            instant=instant,
            object_names=object_names,
        )
        for index, (timestamp_s, caption, observations) in enumerate(frames)
    )
    return ReplayScenario(
        scenario_id=scenario_id,
        title=title,
        summary=summary,
        focus_object_id=focus_object_id,
        duration_s=duration_s,
        frames=replay_frames,
        key_events=tuple(key_events),
    )


def _mug_initial() -> ReplayMapEntry:
    return _map_entry("mug", "Mug", "mug", (0.0, 0.0, 0.80))


def _bottle() -> ReplayMapEntry:
    return _map_entry("bottle", "Bottle", "bottle", (0.45, 0.30, 0.90))


def _table() -> ReplayMapEntry:
    return _map_entry("table", "Table", "table", (0.15, 0.10, 0.40))


def build_passing_person() -> ReplayScenario:
    """A person crosses the view. EgoPersist rejects the human candidate."""
    initial_map = (_mug_initial(), _bottle(), _table())
    names = {
        "mug": "Mug",
        "bottle": "Bottle",
        "table": "Table",
        "person": "Passing person",
    }
    observations: list[tuple[float, str, tuple[ReplayObservation, ...]]] = []
    person_y = (-0.35, -0.20, -0.05, 0.10, 0.25, 0.40)
    captions = (
        "Baseline map. The mug is on the table.",
        "A person enters from the left.",
        "The person crosses the mug area.",
        "The person is close to the mug.",
        "The person leaves to the right.",
        "The person is gone. The mug is in the same place.",
    )
    for index, y_m in enumerate(person_y):
        timestamp_s = index * KEYFRAME_S
        person = _observation(
            "person",
            "Passing person",
            "person",
            (-0.22 + 0.10 * index, y_m, 0.90),
            (0.26, 0.26, 1.70),
            confidence=0.88,
            human_caused=True,
        )
        observations.append(
            (
                timestamp_s,
                captions[index],
                (
                    _observation(
                        "mug", "Mug", "mug", (0.0, 0.0, 0.80), (0.10, 0.10, 0.12)
                    ),
                    _observation(
                        "bottle",
                        "Bottle",
                        "bottle",
                        (0.45, 0.30, 0.90),
                        (0.08, 0.08, 0.28),
                    ),
                    _observation(
                        "table",
                        "Table",
                        "table",
                        (0.15, 0.10, 0.40),
                        (0.80, 0.60, 0.05),
                    ),
                    person,
                ),
            )
        )
    events = (
        ReplayEvent(
            timestamp_s=0.0,
            title="Map ready",
            detail="The mug is committed at (0.00, 0.00, 0.80) m.",
            kind="stable",
        ),
        ReplayEvent(
            timestamp_s=0.5,
            title="Person detected",
            detail="The candidate has a human cause. Do not add it to the map.",
            kind="reject",
        ),
        ReplayEvent(
            timestamp_s=2.5,
            title="Mug unchanged",
            detail="The mug observation stays inside the 0.10 m change gate.",
            kind="stable",
        ),
        ReplayEvent(
            timestamp_s=3.0,
            title="Map protected",
            detail="The passing person does not cause a commit.",
            kind="reject",
        ),
    )
    return _run_scenario(
        scenario_id="passing_person",
        title="Passing person",
        summary=(
            "A human crosses the view. Instant update inserts a false object. "
            "EgoPersist rejects the candidate and keeps the mug."
        ),
        focus_object_id="mug",
        duration_s=3.0,
        initial_map=initial_map,
        frames=observations,
        key_events=events,
        object_names=names,
    )


def build_durable_move() -> ReplayScenario:
    """A mug moves and stays. EgoPersist commits after the gate."""
    initial_map = (_mug_initial(), _bottle(), _table())
    names = {"mug": "Mug", "bottle": "Bottle", "table": "Table"}
    frames: list[tuple[float, str, tuple[ReplayObservation, ...]]] = []
    for index in range(19):
        timestamp_s = index * KEYFRAME_S
        if timestamp_s < 2.5:
            mug_position: Position3 = (0.0, 0.0, 0.80)
            caption = "The mug is stable in its committed position."
        else:
            mug_position = (0.42, 0.05, 0.80)
            if timestamp_s == 2.5:
                caption = "A hand moves the mug. The first changed keyframe defers."
            else:
                caption = "The mug remains in the new position. Count support."
        frames.append(
            (
                timestamp_s,
                caption,
                (
                    _observation("mug", "Mug", "mug", mug_position, (0.10, 0.10, 0.12)),
                    _observation(
                        "bottle",
                        "Bottle",
                        "bottle",
                        (0.45, 0.30, 0.90),
                        (0.08, 0.08, 0.28),
                    ),
                    _observation(
                        "table",
                        "Table",
                        "table",
                        (0.15, 0.10, 0.40),
                        (0.80, 0.60, 0.05),
                    ),
                ),
            )
        )
    events = (
        ReplayEvent(
            timestamp_s=0.0,
            title="Mug committed",
            detail="The mug starts at (0.00, 0.00, 0.80) m.",
            kind="stable",
        ),
        ReplayEvent(
            timestamp_s=2.5,
            title="Move detected",
            detail="The pose change is above 0.10 m. The map defers the change.",
            kind="defer",
        ),
        ReplayEvent(
            timestamp_s=7.5,
            title="Commit gate passes",
            detail="The new position has 11 visible keyframes and 5.0 s dwell.",
            kind="commit",
        ),
    )
    return _run_scenario(
        scenario_id="durable_mug_move",
        title="Durable mug move",
        summary=(
            "The mug moves and remains. Instant update commits immediately. "
            "EgoPersist waits for five visible keyframes and five seconds."
        ),
        focus_object_id="mug",
        duration_s=9.0,
        initial_map=initial_map,
        frames=frames,
        key_events=events,
        object_names=names,
    )


def build_occlusion() -> ReplayScenario:
    """A mug is covered and later revealed. EgoPersist keeps the map entry."""
    initial_map = (_mug_initial(), _bottle(), _table())
    names = {"mug": "Mug", "bottle": "Bottle", "table": "Table"}
    frames: list[tuple[float, str, tuple[ReplayObservation, ...]]] = []
    for index in range(14):
        timestamp_s = index * KEYFRAME_S
        if timestamp_s < 2.0:
            mug_visible = True
            caption = "The mug is visible and stable."
        elif timestamp_s < 5.0:
            mug_visible = False
            caption = "A tray covers the mug. EgoPersist defers removal."
        else:
            mug_visible = True
            caption = "The mug is visible again in the committed position."
        items = [
            _observation(
                "bottle", "Bottle", "bottle", (0.45, 0.30, 0.90), (0.08, 0.08, 0.28)
            ),
            _observation(
                "table", "Table", "table", (0.15, 0.10, 0.40), (0.80, 0.60, 0.05)
            ),
        ]
        if mug_visible:
            items.append(
                _observation("mug", "Mug", "mug", (0.0, 0.0, 0.80), (0.10, 0.10, 0.12))
            )
        frames.append((timestamp_s, caption, tuple(items)))
    events = (
        ReplayEvent(
            timestamp_s=0.0,
            title="Mug visible",
            detail="The mug is committed at (0.00, 0.00, 0.80) m.",
            kind="stable",
        ),
        ReplayEvent(
            timestamp_s=2.0,
            title="Occlusion starts",
            detail="No mug mask is available. Non-observation is not removal evidence.",
            kind="defer",
        ),
        ReplayEvent(
            timestamp_s=5.0,
            title="Mug reappears",
            detail="The observation agrees with the kept map entry.",
            kind="stable",
        ),
    )
    return _run_scenario(
        scenario_id="occlusion",
        title="Occlusion",
        summary=(
            "A tray covers the mug for three seconds. Instant update removes the mug. "
            "EgoPersist defers and keeps the committed entry."
        ),
        focus_object_id="mug",
        duration_s=6.5,
        initial_map=initial_map,
        frames=frames,
        key_events=events,
        object_names=names,
    )


def build_demo_replay() -> ReplayDocument:
    """Return the complete, deterministic demo replay."""
    return ReplayDocument(
        schema_version=SCHEMA_VERSION,
        title="EgoPersist synthetic replay",
        scenarios=(
            build_passing_person(),
            build_durable_move(),
            build_occlusion(),
        ),
    )
