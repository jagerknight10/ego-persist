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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


def _split_ids(path: Path, name: str) -> list[str]:
    """Return the sequence list for one split, alias-tolerant."""
    from egopersist.data.splits import normalize_split_raw

    raw = json.loads(Path(path).read_text())
    return normalize_split_raw(raw).get(name, [])


def _tracks_from(seq_dir: Path) -> dict[str, list[dict]]:
    """Group cached feature rows by track, in file order."""
    out: dict[str, list[dict]] = {}
    for line in _read_lines(seq_dir / "track_features.jsonl"):
        row = json.loads(line)
        out.setdefault(row["track_id"], []).append(row)
    return out


def _track_labels_from(seq_dir: Path) -> dict[str, list[str]]:
    """Group cached label rows by track, in file order."""
    out: dict[str, list[str]] = {}
    for line in _read_lines(seq_dir / "track_labels.jsonl"):
        row = json.loads(line)
        out.setdefault(row["track_id"], []).append(row["label"])
    return out


def _stage_run_id() -> str:
    """Cluster run id: stage name, with the seed for train arrays."""
    seed = os.environ.get("EGOPERSIST_SEED", "")
    stage = _stage_from_argv()
    if stage == "train" and seed:
        return f"train_seed_{seed}"
    return stage


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
            labels[row["track_id"]] = [LabelClass(item) for item in row["labels"]]
        meta: dict[str, str] = {}
        for line in _read_lines(seq_dir / "track_meta.jsonl"):
            row = json.loads(line)
            meta[row["track_id"]] = row["motion_type"]
        yield sequence_id, observations, labels, meta


def run_features(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="egopersist.evidence.cli")
    parser.add_argument("--stage", required=True, choices=["features", "train"])
    parser.parse_args(list(argv))

    run_root = Path(_env("EGOPERSIST_RUN_ROOT"))
    run_id = os.environ.get("EGOPERSIST_RUN_ID", "") or _stage_run_id()
    run_dir = run_root / run_id
    splits_path = Path(_env("EGOPERSIST_SPLIT_FILE"))
    sequence_ids = sorted(
        set(_split_ids(splits_path, "train")) | set(_split_ids(splits_path, "val"))
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
                                key.value: value for key, value in probs.items()
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
                "run_id": run_id,
                "input_dim": INPUT_DIM,
                "tracks": n_tracks,
                "steps": n_steps,
            }
        )
    )
    return 0


def _load_bundle(run_dir: Path, sequence_ids: list[str], split: str) -> Any | None:
    """Pack one split into a TrainBundle (torch tensors)."""
    import torch

    from egopersist.evidence.model import TrainBundle, class_index

    feats: list[list[float]] = []
    labs: list[list[int]] = []
    masks: list[list[bool]] = []
    tids: list[str] = []
    for sequence_id in sequence_ids:
        seq_dir = run_dir / "features" / sequence_id
        features_by_track = _tracks_from(seq_dir)
        labels_by_track = {
            track_id: [class_index(LabelClass(item)) for item in items]
            for track_id, items in _track_labels_from(seq_dir).items()
        }
        for track_id in sorted(features_by_track):
            items = sorted(features_by_track[track_id], key=lambda r: r["timestamp_s"])
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
    Y = torch.full(
        (len(feats), max_len), class_index(LabelClass.UNKNOWN), dtype=torch.int64
    )
    M = torch.zeros(len(feats), max_len, dtype=torch.bool)
    for i, (row, labels, mask) in enumerate(zip(feats, labs, masks, strict=False)):
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

    from egopersist.evidence.model import train_model

    run_root = Path(_env("EGOPERSIST_RUN_ROOT"))
    run_id = os.environ.get("EGOPERSIST_RUN_ID", "") or _stage_run_id()
    run_dir = run_root / run_id
    # Feature cache lives in the base experiment run, shared by all seeds.
    base_run_id = os.environ.get("EGOPERSIST_BASE_RUN_ID", run_id)
    base_dir = run_root / base_run_id
    model_config = _load_yaml(_env("EGOPERSIST_MODEL_CONFIG"))
    seed = int(os.environ.get("EGOPERSIST_SEED", "0"))
    splits_path = Path(_env("EGOPERSIST_SPLIT_FILE"))

    bundles = []
    for split in ("train", "val"):
        bundle = _load_bundle(base_dir, _split_ids(splits_path, split), split)
        if bundle is not None:
            bundles.append(bundle)
    if not bundles:
        raise SystemExit("no training data found; run the features stage first")

    out_dir = run_dir / "train" / f"seed_{seed}"
    train_model(bundles, model_config, out_dir, seed)

    if os.environ.get("EGOPERSIST_MODEL_STAGE") == "1":
        import torch

        from egopersist.contracts import LabelClass
        from egopersist.evidence.calibrate import fit_temperature
        from egopersist.evidence.model import (
            build_model,
            class_index,
            predict_probabilities,
        )

        model = build_model(model_config)
        state = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)

        # Temperature scaling fitted on the validation split only.
        val_sequences = (
            _split_ids(splits_path, "val") or _split_ids(splits_path, "train")[:1]
        )
        val_logits: list[list[float]] = []
        val_targets: list[int] = []
        temperature = 1.0
        for sequence_id in val_sequences:
            seq_dir = base_dir / "features" / sequence_id
            features_by_track = _tracks_from(seq_dir)
            labels_by_track = _track_labels_from(seq_dir)
            for track_id, rows in sorted(features_by_track.items()):
                labels = labels_by_track.get(track_id, [])
                n = min(len(rows), len(labels))
                if n < 2:
                    continue
                x = torch.tensor(
                    [row["values"] for row in rows[:n]], dtype=torch.float32
                )
                logits, _ = model(x.unsqueeze(0))
                for k in range(n):
                    val_logits.append(logits[0, k].tolist())
                    val_targets.append(class_index(LabelClass(labels[k])))
        if val_logits:
            temperature = fit_temperature(val_logits, val_targets)
        (out_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "stage": "model",
                    "temperature": temperature,
                    "val_n": len(val_logits),
                },
                sort_keys=True,
            )
            + "\n"
        )

        # Model probability traces for the eval stage. Written under this
        # seed run: model_probs/<sequence_id>/track_probabilities_model.jsonl.
        # The eval stage takes EGOPERSIST_MODEL_RUN_ID to select a seed.
        probs_root = run_dir / "model_probs"
        for split_name in ("train", "val"):
            for sequence_id in _split_ids(splits_path, split_name):
                seq_dir = base_dir / "features" / sequence_id
                lines: list[str] = []
                for track_id, rows in sorted(_tracks_from(seq_dir).items()):
                    if not rows:
                        continue
                    x = torch.tensor(
                        [row["values"] for row in rows], dtype=torch.float32
                    )
                    mask = torch.ones(len(rows), dtype=torch.bool)
                    probs = predict_probabilities(
                        model, x.unsqueeze(0), mask, temperature
                    )
                    for row, prob_map in zip(rows, probs, strict=False):
                        lines.append(
                            json.dumps(
                                {
                                    "track_id": track_id,
                                    "timestamp_s": row["timestamp_s"],
                                    "probabilities": {
                                        label.value: float(value)
                                        for label, value in prob_map.items()
                                    },
                                },
                                sort_keys=True,
                            )
                        )
                target = probs_root / sequence_id
                target.mkdir(parents=True, exist_ok=True)
                (target / "track_probabilities_model.jsonl").write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
        (out_dir / "model_stage.json").write_text(
            json.dumps(
                {
                    "stage": "model",
                    "run_id": run_id,
                    "temperature": temperature,
                },
                sort_keys=True,
            )
            + "\n"
        )

    print(
        json.dumps(
            {
                "stage": "train",
                "run_id": run_id,
                "seed": seed,
                "out": str(out_dir),
            }
        )
    )
    return 0


def _stage_from_argv(argv: Sequence[str] | None = None) -> str:
    tokens = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(tokens):
        if token == "--stage" and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith("--stage="):
            return token.split("=", 1)[1]
    return "features"


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = _stage_from_argv(argv)
    if stage == "features":
        return run_features(argv)
    if stage == "train":
        return run_train(argv)
    print(json.dumps({"error": f"unknown stage {stage}"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
