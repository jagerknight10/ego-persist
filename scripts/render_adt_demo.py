#!/usr/bin/env python3
"""Render a short, real ADT RGB decision video on the cluster."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from wise_egomap.data.adt import load_adt_sequence
from wise_egomap.data.vrs import AriaVrsReader, find_vrs_file
from wise_egomap.visualization.adt_video import (
    boxes_near_time,
    load_decision_traces,
    render_frame,
    select_demo_start_s,
)


def parse_args() -> argparse.Namespace:
    """Return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--eval-run-id", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--duration-s", type=float, default=16.0)
    parser.add_argument("--fps", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    """Render RGB frames, then encode a web-compatible MP4."""
    args = parse_args()
    if args.duration_s <= 0.0 or args.fps <= 0:
        raise SystemExit("duration and fps must be positive")
    suffix = args.output.suffix.lower()
    if suffix not in {".mp4", ".gif"}:
        raise SystemExit("output must end in .mp4 or .gif")
    if suffix == ".mp4" and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required for MP4 output; use .gif as a fallback")

    sequence_dir = args.data_root / args.sequence
    sequence = load_adt_sequence(sequence_dir)
    start_s = (
        args.start_s
        if args.start_s is not None
        else select_demo_start_s(sequence, args.duration_s)
    )
    end_s = min(start_s + args.duration_s, sequence.end_s)
    if end_s <= start_s:
        raise SystemExit("selected video interval is outside the annotated sequence")

    reader = AriaVrsReader(find_vrs_file(sequence_dir / "main_vrs"))
    decisions = load_decision_traces(
        args.run_root
        / args.eval_run_id
        / "eval"
        / "decisions.jsonl"
    )
    rgb_boxes = tuple(
        sorted(
            (box for box in sequence.boxes_2d if box.stream_id == "214-1"),
            key=lambda box: box.timestamp_s,
        )
    )
    frame_count = int(round((end_s - start_s) * args.fps))
    with tempfile.TemporaryDirectory(prefix="wise-egomap-video-") as temporary:
        frame_dir = Path(temporary)
        for index in range(frame_count):
            timestamp_s = start_s + index / args.fps
            image = reader.image_at("camera-rgb", timestamp_s)
            render_frame(
                image=image,
                boxes=boxes_near_time(rgb_boxes, timestamp_s),
                traces=decisions,
                timestamp_s=timestamp_s,
                output_path=frame_dir / f"frame-{index:05d}.png",
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".mp4":
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(args.fps),
                    "-i",
                    str(frame_dir / "frame-%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(args.output),
                ],
                check=True,
            )
        else:
            from PIL import Image

            frames = [
                Image.open(frame_dir / f"frame-{index:05d}.png")
                for index in range(frame_count)
            ]
            frames[0].save(
                args.output,
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / args.fps),
                loop=0,
            )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
