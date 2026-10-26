"""Cluster-side renderer for real ADT RGB-D decision videos."""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wise_egomap.data.adt import AdtSequence, Box2D


@dataclass(frozen=True)
class DecisionTrace:
    """One evaluated decision used only for a video overlay."""

    timestamp_s: float
    action: str
    label: str
    confidence: float
    probabilities: dict[str, float]


def load_decision_traces(path: str | Path) -> dict[str, tuple[DecisionTrace, ...]]:
    """Load per-track decision traces from an evaluation JSONL file."""
    grouped: dict[str, list[DecisionTrace]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        grouped[str(item["track_id"])].append(
            DecisionTrace(
                timestamp_s=float(item["timestamp_s"]),
                action=str(item["action"]),
                label=str(item["label"]),
                confidence=float(item["confidence"]),
                probabilities={
                    str(name): float(value)
                    for name, value in dict(item["probabilities"]).items()
                },
            )
        )
    return {
        track_id: tuple(sorted(values, key=lambda value: value.timestamp_s))
        for track_id, values in grouped.items()
    }


def nearest_decision(
    trace: Iterable[DecisionTrace], timestamp_s: float, tolerance_s: float = 0.6
) -> DecisionTrace | None:
    """Return the nearest causal decision when it is within the tolerance."""
    values = tuple(trace)
    if not values:
        return None
    times = [value.timestamp_s for value in values]
    index = bisect.bisect_left(times, timestamp_s)
    candidates = values[max(0, index - 1) : index + 1]
    best = min(candidates, key=lambda value: abs(value.timestamp_s - timestamp_s))
    return best if abs(best.timestamp_s - timestamp_s) <= tolerance_s else None


def boxes_near_time(
    boxes: Iterable[Box2D], timestamp_s: float, tolerance_s: float = 0.6
) -> tuple[Box2D, ...]:
    """Return boxes near a timestamp; input boxes must be in time order."""
    values = tuple(boxes)
    times = [box.timestamp_s for box in values]
    index = bisect.bisect_left(times, timestamp_s)
    start = index
    while start and timestamp_s - times[start - 1] <= tolerance_s:
        start -= 1
    end = index
    while end < len(values) and times[end] - timestamp_s <= tolerance_s:
        end += 1
    return tuple(
        sorted(
            (
                box
                for box in values[start:end]
                if abs(box.timestamp_s - timestamp_s) <= tolerance_s
            ),
            key=lambda box: box.area_px,
            reverse=True,
        )
    )


def select_demo_start_s(sequence: AdtSequence, duration_s: float) -> float:
    """Choose an annotated RGB interval with a dynamic object when available."""
    dynamic_ids = set(sequence.dynamic_instance_ids())
    for box in sequence.boxes_2d:
        if box.stream_id == "214-1" and box.object_uid in dynamic_ids:
            return min(
                max(box.timestamp_s - 2.0, sequence.start_s),
                sequence.end_s - duration_s,
            )
    return min(sequence.start_s + 2.0, sequence.end_s - duration_s)


def action_color(action: str) -> str:
    """Return a fixed overlay colour for a map action."""
    return {
        "keep": "#31c48d",
        "defer": "#f6c344",
        "commit": "#ff7a59",
        "reject": "#8aa3ff",
        "revert": "#df75d3",
    }.get(action, "#e8eef0")


def render_frame(
    *,
    image: Any,
    boxes: tuple[Box2D, ...],
    traces: dict[str, tuple[DecisionTrace, ...]],
    timestamp_s: float,
    output_path: Path,
    max_boxes: int = 8,
) -> None:
    """Write one annotated RGB frame as a PNG image."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "rendering needs matplotlib (install the demo extra)"
        ) from exc

    figure, axes = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.9, 0.8]}
    )
    figure.patch.set_facecolor("#10191d")
    image_ax, state_ax = axes
    image_ax.imshow(image)
    image_ax.set_axis_off()
    image_ax.set_title("Egocentric RGB observation", color="white", loc="left")
    state_ax.set_facecolor("#17262b")
    state_ax.tick_params(colors="#c7d5d7")
    state_ax.set_title("Causal map state", color="white", loc="left")
    state_ax.set_xlim(0.0, 1.0)
    state_ax.set_ylim(0.0, 1.0)
    state_ax.set_xticks([])
    state_ax.set_yticks([])

    selected = boxes[:max_boxes]
    for row, box in enumerate(selected):
        trace = nearest_decision(traces.get(str(box.object_uid), ()), timestamp_s)
        action = trace.action if trace else "observe"
        colour = action_color(action)
        image_ax.add_patch(
            Rectangle(
                (box.x_min_px, box.y_min_px),
                box.x_max_px - box.x_min_px,
                box.y_max_px - box.y_min_px,
                fill=False,
                edgecolor=colour,
                linewidth=2.0,
            )
        )
        title = f"{box.object_uid} · {action.upper()}"
        image_ax.text(
            box.x_min_px,
            max(4.0, box.y_min_px - 5.0),
            title,
            color="white",
            fontsize=7,
            bbox={"facecolor": colour, "edgecolor": "none", "pad": 2.5},
        )
        y = 0.91 - 0.105 * row
        state_ax.text(0.08, y, f"track {box.object_uid}", color="white", fontsize=8)
        state_ax.text(
            0.08, y - 0.036, action.upper(), color=colour, fontsize=9, weight="bold"
        )
        if trace:
            state_ax.text(
                0.08,
                y - 0.071,
                f"{trace.label}  ·  p={trace.confidence:.2f}",
                color="#b9c8ca",
                fontsize=7,
            )

    state_ax.text(0.08, 0.05, f"t = {timestamp_s:.2f} s", color="#b9c8ca", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=130,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)
