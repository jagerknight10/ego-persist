"""Evidence stage CLI (cluster entry point).

``EGOPERSIST_FEATURES_MODULE`` and ``EGOPERSIST_TRAIN_MODULE`` both point
here. Stage ``features`` is CPU-only: it reads the preprocess output,
computes the 20 causal features per keyframe, runs the Bayes
accumulator, and writes a feature cache plus per-track probability
traces. Stage ``train`` (cluster, torch) trains the evidence GRU on the
cached features and writes a checkpoint with training summary.

Environment:
    EGOPERSIST_RUN_ROOT, EGOPERSIST_RUN_ID, EGOPERSIST_MODEL_CONFIG,
    EGOPERSIST_SPLIT_FILE, EGOPERSIST_SEED.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from egopersist.contracts import LabelClass, Observation, TrackObservation
from egopersist.evidence.bayes import BayesAccumulator
from egopersist.evidence.features import (
    FEATURE_NAMES,
    INPUT_DIM,
    TrackFeatureState,
)


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"environment variable {name} is not set")
    return value


def _load_yaml(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line]


def _iter_preprocessed(run_dir: Path, sequence_ids: list[str]):
    """Yield (sequence_id, observations, labels, meta) per sequence.

    ``observations`` maps track_id -> list[Observation] in time order.
    ``labels`` maps track_id -> list[LabelClass], one entry per
    observation in the same order (the preprocess stage writes the
    label list over the track's visible keyframes, in order).
    ``meta`` maps track_id -> motion_type.
    """
    for sequence_id in sequence_ids:
        seq_dir = run_dir / "preprocess" / sequence_id
        observations: dict[str, list[Observation]] = {}
        for line in _read_lines(seq_dir / "observations.jsonl"):
            obs = Observation.from_json(line)
            observations.setdefault(obs.instance_id, []).append(obs)
        for track in observations.values():
            track.sort(key=lambda o: o.timestamp_s)
        labels: dict[str, list[LabelClass]] = {}
        for line in _read_lines(seq_dir / "track_labels.jsonl"):
            row = json.loads(line)
            labels[row["track_id"]] = [
                LabelClass(item) for item in row["labels"]
            ]
        meta: dict[str, str] = {}
        for line in _read_lines(seq_dir / "track_meta.jsonl"):
            row = json.loads(line)
            meta[row["track_id"]] = row["motion_type"]
        yield sequence_id, observations, labels, meta


def run_features(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="egopersist.evidence.cli")
    parser.add_argument("--stage", required=True, choices=["features", "train"])
    args = parser.parse_args(list(argv))

    run_dir = Path(_env("EGOPERSIST_RUN_ROOT")) / _env("EGOPERSIST_RUN_ID")
    splits_raw = json.loads(Path(_env("EGOPERSIST_SPLIT_FILE")).read_text())
    sequence_ids = sorted(
        {item for key in ("train", "val") for item in splits_raw.get(key, [])}
    )
    out_dir = run_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tracks = 0
    n_steps = 0
    for sequence_id, observations, labels, meta in _iter_preprocessed(
        run_dir, sequence_ids
    ):
        seq_out = out_dir / sequence_id
        seq_out.mkdir(parents=True, exist_ok=True)
        feature_lines: list[str] = []
        prob_lines: list[str] = []
        label_lines: list[str] = []
        for track_id in sorted(observations):
            obs_list = observations[track_id]
            if not obs_list:
                continue
            # Frozen-map committed pose: the track's first observed pose
            # is the pre-change reference the map holds. Causal: uses
            # only the past.
            anchor = tuple(obs_list[0].centroid_world_m)
            dynamic = meta.get(track_id, "static") == "dynamic"
            track_labels = labels.get(track_id, [])
            state = TrackFeatureState()
            bayes = BayesAccumulator(window=5)
            for index, obs in enumerate(obs_list):
                view = TrackObservation(
                    track_id=track_id,
                    observation=obs,
                    committed_position_m=anchor,  # type: ignore[arg-type]
                )
                values = state.update(view, dynamic)
                bayes.observe(values, dynamic)
                probs = bayes.probabilities()
                feature_lines.append(
                    json.dumps(
                        {
                            "sequence_id": sequence_id,
                            "track_id": track_id,
                            "timestamp_s": obs.timestamp_s,
                            "frame_id": obs.frame_id,
                            "values": list(values),
                        },
                        sort_keys=True,
                    )
                )
                prob_lines.append(
                    json.dumps(
                        {
                            "track_id": track_id,
                            "timestamp_s": obs.timestamp_s,
                            "probabilities": {
                                key.value: value
                                for key, value in probs.items()
                            },
                        },
                        sort_keys=True,
                    )
                )
                if index < len(track_labels):
                    label_lines.append(
                        json.dumps(
                            {
                                "track_id": track_id,
                                "timestamp_s": obs.timestamp_s,
                                "label": track_labels[index].value,
                            },
                            sort_keys=True,
                        )
                    )
            n_tracks += 1
            n_steps += len(obs_list)
        (seq_out / "track_features.jsonl").write_text(
            "\n".join(feature_lines) + ("\n" if feature_lines else "")
        )
        (seq_out / "track_probabilities_bayes.jsonl").write_text(
            "\n".join(prob_lines) + ("\n" if prob_lines else "")
        )
        (seq_out / "track_labels.jsonl").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else "")
        )

    (out_dir / "feature_names.json").write_text(
        json.dumps(list(FEATURE_NAMES), indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "stage": "features",
                "run_id": _env("EGOPERSIST_RUN_ID"),
                "input_dim": INPUT_DIM,
                "tracks": n_tracks,
                "steps": n_steps,
            }
        )
    )
    return 0


def _load_bundle(
    run_dir: Path, sequence_ids: list[str], split: str
) -> "Any | None":
    """Pack one split into a TrainBundle (torch tensors)."""
    import torch

    from egopersist.evidence.model import CLASS_ORDER, TrainBundle, class_index

    feats: list[list[float]] = []
    labs: list[list[int]] = []
    masks: list[list[bool]] = []
    tids: list[str] = []
    for sequence_id in sequence_ids:
        seq_dir = run_dir / "features" / sequence_id
        features_by_track: dict[str, list[dict]] = {}
        for line in _read_lines(seq_dir / "track_features.jsonl"):
            row = json.loads(line)
            features_by_track.setdefault(row["track_id"], []).append(row)
        labels_by_track: dict[str, list[int]] = {}
        for line in _read_lines(seq_dir / "track_labels.jsonl"):
            row = json.loads(line)
            labels_by_track.setdefault(row["track_id"], []).append(
                class_index(LabelClass(row["label"]))
            )
        for track_id in sorted(features_by_track):
            items = sorted(
                features_by_track[track_id], key=lambda r: r["timestamp_s"]
            )
            labels = labels_by_track.get(track_id, [])
            n = min(len(items), len(labels))
            if n == 0:
                continue
            feats.append([row["values"] for row in items[:n]])
            labs.append(labels[:n])
            masks.append([True] * n)
            tids.append(track_id)
    if not feats:
        return None
    max_len = max(len(row) for row in feats)
    X = torch.zeros(len(feats), max_len, INPUT_DIM)
    Y = torch.full((len(feats), max_len), class_index(LabelClass.UNKNOWN), dtype=torch.int64)
    M = torch.zeros(len(feats), max_len, dtype=torch.bool)
    for i, (row, labels, mask) in enumerate(zip(feats, labs, masks)):
        X[i, : len(row)] = torch.tensor(row)
        Y[i, : len(labels)] = torch.tensor(labels)
        M[i, : len(mask)] = torch.tensor(mask)
    return TrainBundle(
        features=X,
        labels=Y,
        masks=M,
        track_ids=tuple(tids),
        sequence_ids=tuple(sequence_ids),
        split=split,
    )


def run_train(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="egopersist.evidence.cli")
    parser.add_argument("--stage", required=True, choices=["features", "train"])
    parser.parse_args(list(argv))

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("train stage requires torch (run on the cluster)") from exc

    from egopersist.evidence.model import load_model_config, train_model

    run_dir = Path(_env("EGOPERSIST_RUN_ROOT")) / _env("EGOPERSIST_RUN_ID")
    model_config = _load_yaml(_env("EGOPERSIST_MODEL_CONFIG"))
    seed = int(os.environ.get("EGOPERSIST_SEED", "0"))
    splits_raw = json.loads(Path(_env("EGOPERSIST_SPLIT_FILE")).read_text())

    bundles = []
    for split in ("train", "val"):
        bundle = _load_bundle(
            run_dir, splits_raw.get(split, []), split
        )
        if bundle is not None:
            bundles.append(bundle)
    if not bundles:
        raise SystemExit("no training data found; run the features stage first")

    out_dir = run_dir / "train" / f"seed_{seed}"
    train_model(bundles, model_config, out_dir, seed)
    print(
        json.dumps(
            {
                "stage": "train",
                "run_id": _env("EGOPERSIST_RUN_ID"),
                "seed": seed,
                "out": str(out_dir),
            }
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = "features"
    for i, token in enumerate(argv):
        if token == "--stage" and i + 1 < len(argv):
            stage = argv[i + 1]
        elif token.startswith("--stage="):
            stage = token.split("=", 1)[1]
    if stage == "features":
        return run_features(argv)
    if stage == "train":
        return run_train(argv)
    print(json.dumps({"error": f"unknown stage {stage}"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
