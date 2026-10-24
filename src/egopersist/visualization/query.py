"""Natural-language object query for the synthetic demo."""

from __future__ import annotations

from typing import Sequence

from egopersist.contracts import LabelClass
from egopersist.visualization.replay import ReplayMapEntry, ReplayQuery

STATE_TEXT = {
    LabelClass.STABLE: "stable",
    LabelClass.TRANSIENT: "transient",
    LabelClass.PERSISTENT_CHANGE: "persistent change",
    LabelClass.UNKNOWN: "not fully visible",
}


def answer_object_query(
    object_name: str,
    map_entries: Sequence[ReplayMapEntry],
    *,
    timestamp_s: float,
) -> ReplayQuery:
    """Answer "Where is the object?" from the current map.

    The answer uses only the map at the requested time. Missing map data gives
    an explicit no-match answer instead of a guessed position.
    """
    if not object_name:
        raise ValueError("object_name must be non-empty")
    if timestamp_s < 0.0:
        raise ValueError("timestamp_s must be non-negative")
    target = object_name.casefold()
    matches = [
        entry
        for entry in map_entries
        if target in entry.object_id.casefold()
        or target in entry.display_name.casefold()
        or target in entry.kind.casefold()
    ]
    if not matches:
        raise LookupError(f"object not found in map: {object_name}")
    entry = sorted(matches, key=lambda item: item.object_id)[0]
    x_m, y_m, z_m = entry.position_m
    state_text = STATE_TEXT[entry.state]
    if entry.state is LabelClass.UNKNOWN:
        answer = (
            f"{entry.display_name} is {state_text}. The current map keeps it at "
            f"({x_m:.2f}, {y_m:.2f}, {z_m:.2f}) m with "
            f"{entry.confidence:.0%} confidence."
        )
    elif entry.state is LabelClass.PERSISTENT_CHANGE:
        answer = (
            f"{entry.display_name} is at ({x_m:.2f}, {y_m:.2f}, {z_m:.2f}) m. "
            f"The map has a committed change with {entry.confidence:.0%} confidence."
        )
    else:
        answer = (
            f"{entry.display_name} is at ({x_m:.2f}, {y_m:.2f}, {z_m:.2f}) m. "
            f"Map state is {state_text} with {entry.confidence:.0%} confidence."
        )
    return ReplayQuery(
        object_id=entry.object_id,
        object_name=entry.display_name,
        state=entry.state,
        position_m=entry.position_m,
        confidence=entry.confidence,
        answer=answer,
    )
