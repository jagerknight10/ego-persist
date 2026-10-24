"""CPU video renderer for the synthetic replay.

The renderer uses Matplotlib primitives only. It does not read video, masks, or
network resources. Use an MP4 writer on the cluster, or a GIF writer for a
small local artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from wise_egomap.contracts import DecisionAction, LabelClass
from wise_egomap.visualization.replay import (
    ReplayDocument,
    ReplayFrame,
    ReplayMapEntry,
    ReplayScenario,
)

BACKGROUND = "#101418"
PANEL = "#182028"
TEXT = "#eef3f6"
MUTED = "#9fb0bd"
EGO_COLOR = "#57d1c9"
INSTANT_COLOR = "#ffc857"
LINE_COLOR = "#34414c"

STATE_COLORS = {
    LabelClass.STABLE: "#75d787",
    LabelClass.TRANSIENT: "#ff7373",
    LabelClass.PERSISTENT_CHANGE: "#7fb3ff",
    LabelClass.UNKNOWN: "#ffc857",
}


def _new_figure() -> tuple[Any, Any]:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError(
            "Video rendering needs matplotlib. Install the demo extra."
        ) from exc
    figure = Figure(figsize=(12.8, 7.2), dpi=100, facecolor=BACKGROUND)
    FigureCanvasAgg(figure)
    return figure, FigureCanvasAgg


def _draw_map(
    axis: Any,
    entries: Sequence[ReplayMapEntry],
    *,
    title: str,
    color: str,
) -> None:
    from matplotlib.patches import Circle, FancyBboxPatch

    axis.set_facecolor("#0c1115")
    axis.set_xlim(-0.65, 0.8)
    axis.set_ylim(-0.55, 1.25)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(LINE_COLOR)
    axis.set_title(title, color=color, loc="left", fontsize=13, fontweight="bold")
    axis.grid(color="#20303a", linewidth=0.7, alpha=0.6)
    for entry in entries:
        x_m, y_m, _ = entry.position_m
        if entry.kind == "table":
            patch = FancyBboxPatch(
                (x_m - 0.38, y_m - 0.22),
                0.76,
                0.44,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="none",
                alpha=0.20,
            )
        elif entry.kind == "person":
            patch = FancyBboxPatch(
                (x_m - 0.06, y_m - 0.12),
                0.12,
                0.24,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor="none",
                alpha=0.70,
            )
        else:
            radius_m = 0.05 if entry.kind == "mug" else 0.035
            patch = Circle(
                (x_m, y_m),
                radius=radius_m,
                facecolor=color,
                edgecolor="white" if entry.kind == "mug" else "none",
                linewidth=1.0,
                alpha=0.95,
            )
        axis.add_patch(patch)
        axis.text(
            x_m,
            y_m + (0.08 if entry.kind == "mug" else 0.06),
            entry.display_name,
            color=TEXT,
            fontsize=9,
            ha="center",
            va="bottom",
            fontweight="bold" if entry.kind == "mug" else "normal",
        )
        axis.text(
            x_m,
            y_m - (0.11 if entry.kind == "mug" else 0.08),
            entry.state.value.lower().replace("_", " "),
            color=STATE_COLORS[entry.state],
            fontsize=7,
            ha="center",
            va="top",
        )


def _draw_timeline(
    axis: Any,
    scenario: ReplayScenario,
    frame: ReplayFrame,
) -> None:
    axis.set_facecolor(PANEL)
    axis.set_xlim(0.0, scenario.duration_s)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])
    axis.set_xticks([event.timestamp_s for event in scenario.key_events])
    axis.tick_params(axis="x", colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(LINE_COLOR)
    axis.axhline(0.55, color=LINE_COLOR, linewidth=2.0)
    for event in scenario.key_events:
        event_color = {
            "commit": "#7fb3ff",
            "defer": "#ffc857",
            "reject": "#ff7373",
        }.get(event.kind, "#75d787")
        axis.plot(
            [event.timestamp_s, event.timestamp_s],
            [0.43, 0.67],
            color=event_color,
            linewidth=3.0,
        )
        axis.text(
            event.timestamp_s,
            0.78,
            event.title,
            color=MUTED,
            fontsize=7,
            ha="center",
            va="bottom",
        )
    axis.axvline(
        frame.timestamp_s,
        color=EGO_COLOR,
        linewidth=2.0,
        alpha=0.95,
    )
    axis.set_xlabel("time (s)", color=MUTED, fontsize=8)
    axis.set_title("Map timeline", color=MUTED, loc="left", fontsize=9)


def _decision_for(frame: ReplayFrame, object_id: str) -> Any:
    for item in frame.decisions:
        if item.object_id == object_id:
            return item
    return None


def _draw_evidence(axis: Any, frame: ReplayFrame) -> None:
    axis.set_facecolor(PANEL)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(LINE_COLOR)
    axis.set_title("Evidence and decision", color=MUTED, loc="left", fontsize=9)
    items = [
        item for item in frame.evidence if item.object_id == frame.query.object_id
    ][:5]
    y_position = 0.88
    for item in items:
        ratio = min(1.0, item.value / item.max_value)
        threshold = min(1.0, item.threshold / item.max_value)
        color = EGO_COLOR if item.value + 1e-9 >= item.threshold else INSTANT_COLOR
        axis.text(
            0.04,
            y_position,
            f"{item.label}: {item.value:.2f} {item.unit} (gate {item.threshold:.2f})",
            color=TEXT,
            fontsize=8,
            va="top",
        )
        axis.barh(
            y_position - 0.06,
            ratio,
            height=0.025,
            left=0.04,
            color=color,
        )
        axis.plot(
            [0.04 + threshold, 0.04 + threshold],
            [y_position - 0.082, y_position - 0.038],
            color=MUTED,
            linewidth=1.0,
        )
        y_position -= 0.16
    decision = _decision_for(frame, frame.query.object_id)
    action = decision.action.value.upper() if decision is not None else "NONE"
    action_color = {
        DecisionAction.COMMIT: "#7fb3ff",
        DecisionAction.DEFER: "#ffc857",
        DecisionAction.REJECT: "#ff7373",
    }.get(decision.action if decision is not None else DecisionAction.KEEP, "#75d787")
    axis.text(
        0.04,
        0.10,
        f"{action}  {frame.query.object_name}  {frame.query.confidence:.0%} confidence",
        color=action_color,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def render_frame_figure(
    scenario: ReplayScenario,
    frame_index: int,
) -> Any:
    """Return one Matplotlib figure for a scenario keyframe."""
    if frame_index < 0 or frame_index >= len(scenario.frames):
        raise IndexError("frame_index is outside the scenario")
    figure, _ = _new_figure()
    frame = scenario.frames[frame_index]
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(0.8, 3.8, 1.6),
        width_ratios=(1.0, 1.0),
        left=0.04,
        right=0.97,
        top=0.97,
        bottom=0.05,
        hspace=0.42,
        wspace=0.12,
    )
    title_axis = figure.add_subplot(grid[0, :])
    title_axis.set_facecolor(BACKGROUND)
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.80,
        scenario.title,
        color=TEXT,
        fontsize=20,
        fontweight="bold",
        va="top",
    )
    title_axis.text(
        0.0,
        0.30,
        frame.caption,
        color=MUTED,
        fontsize=10,
        va="top",
    )
    title_axis.text(
        1.0,
        0.80,
        f"{frame.timestamp_s:.1f} s",
        color=EGO_COLOR,
        fontsize=13,
        ha="right",
        va="top",
    )
    ego_axis = figure.add_subplot(grid[1, 0])
    instant_axis = figure.add_subplot(grid[1, 1])
    _draw_map(ego_axis, frame.ego_map, title="WISE-EgoMap map", color=EGO_COLOR)
    _draw_map(
        instant_axis,
        frame.instant_map,
        title="Instant update",
        color=INSTANT_COLOR,
    )
    timeline_axis = figure.add_subplot(grid[2, 0])
    _draw_timeline(timeline_axis, scenario, frame)
    evidence_axis = figure.add_subplot(grid[2, 1])
    _draw_evidence(evidence_axis, frame)
    return figure


def write_frame_png(
    scenario: ReplayScenario,
    output_path: str | Path,
    *,
    frame_index: int = 0,
) -> Path:
    """Write one replay frame as a PNG file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = render_frame_figure(scenario, frame_index)
    figure.savefig(path, facecolor=figure.get_facecolor())
    return path


def write_demo_video(
    replay: ReplayDocument,
    output_path: str | Path,
    *,
    fps: int = 2,
    scenario_id: str | None = None,
) -> Path:
    """Render MP4 or GIF video from the synthetic replay.

    MP4 output needs a working FFmpeg binary. GIF output uses Pillow.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in {".mp4", ".gif"}:
        raise ValueError("output_path must end with .mp4 or .gif")
    scenarios = (
        (replay.scenario(scenario_id),) if scenario_id is not None else replay.scenarios
    )
    try:
        from matplotlib import animation
    except ImportError as exc:
        raise RuntimeError(
            "Video rendering needs matplotlib. Install the demo extra."
        ) from exc
    first_figure = render_frame_figure(scenarios[0], 0)
    writer_class = (
        animation.FFMpegWriter if suffix == ".mp4" else animation.PillowWriter
    )
    try:
        writer = writer_class(fps=fps)
        with writer.saving(first_figure, str(path), dpi=100):
            for index, scenario in enumerate(scenarios):
                for frame_index in range(len(scenario.frames)):
                    _figure = (
                        first_figure
                        if index == 0 and frame_index == 0
                        else render_frame_figure(scenario, frame_index)
                    )
                    writer.grab_frame(facecolor=BACKGROUND)
    except Exception as exc:
        if suffix == ".mp4":
            raise RuntimeError(
                "MP4 rendering failed. Check that ffmpeg is available. "
                "Use a .gif output for a small local artifact."
            ) from exc
        raise
    return path
