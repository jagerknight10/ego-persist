"""Command-line interface for the synthetic WISE-EgoMap demo."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from wise_egomap.visualization.query import answer_object_query
from wise_egomap.visualization.replay import ReplayDocument, ReplaySchemaError
from wise_egomap.visualization.scenarios import build_demo_replay
from wise_egomap.visualization.video import write_demo_video, write_frame_png
from wise_egomap.visualization.viewer import write_replay_html, write_replay_json


def _add_replay_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--replay",
        type=Path,
        help="Read an existing replay JSON file. Default: build the demo replay.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the demo command parser."""
    parser = argparse.ArgumentParser(
        prog="wise_egomap",
        description="WISE-EgoMap synthetic demo tools.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate", help="Write replay JSON and a self-contained HTML viewer."
    )
    generate.add_argument("--output", type=Path, default=Path("demo-output"))

    query = commands.add_parser("query", help="Answer the mug query at a replay time.")
    _add_replay_source(query)
    query.add_argument("--scenario", default="durable_mug_move")
    query.add_argument("--time", type=float, default=9.0)
    query.add_argument("--object", default="mug")

    render = commands.add_parser(
        "render", help="Render the replay to a PNG sequence or video."
    )
    _add_replay_source(render)
    render.add_argument("--output", type=Path, default=Path("demo-output/demo.gif"))
    render.add_argument("--fps", type=int, default=2)
    render.add_argument("--scenario")
    render.add_argument("--frame", type=int)
    return parser


def _load_replay(path: Path | None) -> ReplayDocument:
    if path is None:
        return build_demo_replay()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read replay file: {path}") from exc
    return ReplayDocument.from_json(text)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one demo CLI command and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        replay = _load_replay(getattr(args, "replay", None))
        if args.command == "generate":
            output = Path(args.output)
            json_path = write_replay_json(replay, output / "replay.json")
            html_path = write_replay_html(replay, output / "index.html")
            print(f"Wrote {json_path}")
            print(f"Wrote {html_path}")
            return 0
        if args.command == "query":
            scenario = replay.scenario(args.scenario)
            frame = scenario.frame_at(args.time)
            result = answer_object_query(
                args.object,
                frame.ego_map,
                timestamp_s=frame.timestamp_s,
            )
            print(result.answer)
            return 0
        if args.command == "render":
            if args.frame is not None and args.output.suffix.lower() == ".png":
                scenario = replay.scenario(
                    args.scenario or replay.scenarios[0].scenario_id
                )
                path = write_frame_png(
                    scenario,
                    args.output,
                    frame_index=args.frame,
                )
            else:
                path = write_demo_video(
                    replay,
                    args.output,
                    fps=args.fps,
                    scenario_id=args.scenario,
                )
            print(f"Wrote {path}")
            return 0
    except (
        LookupError,
        ReplaySchemaError,
        RuntimeError,
        ValueError,
        IndexError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
