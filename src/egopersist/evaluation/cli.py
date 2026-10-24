"""Evaluation stage CLI (cluster entry point).

``EGOPERSIST_EVAL_MODULE`` points here. Stage ``eval`` replays the causal
state machine over per-keyframe probabilities (from the features stage,
or from a trained model when a checkpoint is present), collects the
Decisions and MapUpdates, scores them against the rule labels, and writes
a machine-readable report plus a RunManifest.

Environment:
    EGOPERSIST_RUN_ROOT, EGOPERSIST_RUN_ID, EGOPERSIST_PROJECT_ROOT,
    EGOPERSIST_LABEL_CONFIG, EGOPERSIST_MODEL_CONFIG, EGOPERSIST_SPLIT_FILE,
    EGOPERSIST_SEED.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from egopersist.contracts import (
    Decision,
    EventLabel,
    LabelClass,
    MapUpdate,
    Observation,
)
from egopersist.evaluation.manifest import build_run_manifest
from egopersist.evaluation.report import SCHEMA_VERSION, evaluation_report
from egopersist.mapping.state_machine import (
    StateMachineConfig,
    TrackStateMachine,
    TrackView,
)


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"environment variable {name} is not set")
    return value


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line]


def _load_yaml(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def _probability_source(
    run_dir: Path, sequence_id: str
) -> tuple[dict[str, dict[float, dict[str, float]]], str]:
    """Return (track -> timestamp -> probs, source_name).

    Prefers the trained model trace when a checkpoint exists; falls back
    to the Bayes trace. The fallback keeps the eval stage runnable on
    CPU with no torch.
    """
    trained = run_dir / "features" / sequence_id / "track_probabilities_model.jsonl"
    if trained.exists():
        out: dict[str, dict[float, dict[str, float]]] = {}
        for line in _read_lines(trained):
            row = json.loads(line)
            out.setdefault(row["track_id"], {})[row["timestamp_s"]] = {
                LabelClass(key): value
                for key, value in row["probabilities"].items()
            }
        return out, "model"
    out = {}
    for line in _read_lines(
        run_dir / "features" / sequence_id / "track_probabilities_bayes.jsonl"
    ):
        row = json.loads(line)
        out.setdefault(row["track_id"], {})[row["timestamp_s"]] = {
            LabelClass(key): value for key, value in row["probabilities"].items()
        }
    return out, "bayes"


def run_eval(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="egopersist.evaluation.cli")
    parser.add_argument("--stage", required=True, choices=["eval"])
    parser.parse_args(list(argv))

    run_root = Path(_env("EGOPERSIST_RUN_ROOT"))
    run_id = _env("EGOPERSIST_RUN_ID")
    run_dir = run_root / run_id
    project_root = _env("EGOPERSIST_PROJECT_ROOT")
    label_config_path = _env("EGOPERSIST_LABEL_CONFIG")
    model_config_path = _env("EGOPERSIST_MODEL_CONFIG")
    split_path = Path(_env("EGOPERSIST_SPLIT_FILE"))

    labels_raw = _load_yaml(label_config_path)
    sm_config = StateMachineConfig.from_labels_yaml(labels_raw)

    splits_raw = json.loads(split_path.read_text())
    sequence_ids = sorted(
        {item for key in ("train", "val") for item in splits_raw.get(key, [])}
    )

    decisions: list[Decision] = []
    updates: list[MapUpdate] = []
    events: list[EventLabel] = []
    per_sequence_source: dict[str, str] = {}

    for sequence_id in sequence_ids:
        obs_by_track: dict[str, list[Observation]] = {}
        for line in _read_lines(
            run_dir / "preprocess" / sequence_id / "observations.jsonl"
        ):
            obs = Observation.from_json(line)
            obs_by_track.setdefault(obs.instance_id, []).append(obs)
        for track in obs_by_track.values():
            track.sort(key=lambda o: o.timestamp_s)

        events.extend(
            EventLabel.from_json(line)
            for line in _read_lines(
                run_dir / "preprocess" / sequence_id / "event_labels.jsonl"
            )
        )

        probabilities, source = _probability_source(run_dir, sequence_id)
        per_sequence_source[sequence_id] = source

        for track_id in sorted(obs_by_track):
            track = obs_by_track[track_id]
            if not track:
                continue
            machine = TrackStateMachine(sm_config)
            anchor = tuple(track[0].centroid_world_m)
            probs_by_ts = probabilities.get(track_id, {})
            meta_line = next(
                (
                    json.loads(line)
                    for line in _read_lines(
                        run_dir / "preprocess" / sequence_id / "track_meta.jsonl"
                    )
                    if json.loads(line)["track_id"] == track_id
                ),
                None,
            )
            dynamic = bool(meta_line and meta_line["motion_type"] == "dynamic")
            for obs in track:
                probs = probs_by_ts.get(obs.timestamp_s)
                if probs is None:
                    continue
                view = TrackView(
                    sequence_id=sequence_id,
                    track_id=track_id,
                    frame_id=obs.frame_id,
                    timestamp_s=obs.timestamp_s,
                    observation=obs,
                    probabilities=probs,
                    committed_position_m=anchor,
                    committed_orientation_wxyz=tuple(
                        obs.orientation_wxyz
                    ),
                    is_static=not dynamic,
                )
                decision, update = machine.update(view)
                decisions.append(decision)
                updates.append(update)

    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = evaluation_report(
        events,
        decisions,
        updates=updates,
        system_id=f"egopersist-{os.environ.get('EGOPERSIST_SEED', '0')}",
    )
    (out_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    (out_dir / "decisions.jsonl").write_text(
        "\n".join(d.to_json() for d in decisions)
        + ("\n" if decisions else "")
    )
    (out_dir / "map_updates.jsonl").write_text(
        "\n".join(u.to_json() for u in updates) + ("\n" if updates else "")
    )

    manifest = build_run_manifest(
        run_id=run_id,
        config_path=model_config_path,
        split_file=split_path,
        label_config_path=label_config_path,
        project_root=project_root,
        extra={
            "schema": SCHEMA_VERSION,
            "probability_source": per_sequence_source,
            "n_decisions": len(decisions),
            "n_updates": len(updates),
            "n_events": len(events),
        },
    )
    (out_dir / "provenance.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, default=str)
        + "\n"
    )
    (out_dir / "RunManifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(
        json.dumps(
            {
                "stage": "eval",
                "run_id": run_id,
                "n_decisions": len(decisions),
                "n_events": len(events),
                "source": per_sequence_source,
            }
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = "eval"
    for i, token in enumerate(argv):
        if token == "--stage" and i + 1 < len(argv):
            stage = argv[i + 1]
        elif token.startswith("--stage="):
            stage = token.split("=", 1)[1]
    if stage == "eval":
        return run_eval(argv)
    print(json.dumps({"error": f"unknown stage {stage}"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
