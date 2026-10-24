"""Evidence GRU: learned 4-label causal track classifier.

Torch is imported lazily inside functions so the module can be imported
on machines without torch (tests, docs, CPU-only pipeline). Training and
inference run on the compute cluster; local runs never train.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from egopersist.contracts import LabelClass
from egopersist.evidence.features import FEATURE_NAMES, INPUT_DIM

CLASS_ORDER: tuple[LabelClass, ...] = (
    LabelClass.STABLE,
    LabelClass.TRANSIENT,
    LabelClass.PERSISTENT_CHANGE,
    LabelClass.UNKNOWN,
)


def class_index(label: LabelClass) -> int:
    return CLASS_ORDER.index(label)


def load_model_config(path: str | Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("version") != 1:
        raise ValueError("unsupported model config version")
    return raw


@dataclass(frozen=True)
class TrainBundle:
    """Packed tensors for one training run (cluster-side only)."""

    features: Any  # torch.Tensor [N, T, D] float32
    labels: Any  # torch.Tensor [N, T] int64
    masks: Any  # torch.Tensor [N, T] bool
    track_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    split: str = "train"


def pack_track(
    feature_rows: Sequence[Sequence[float]],
    label_rows: Sequence[LabelClass | None],
) -> tuple[list[float], list[int], list[bool]]:
    """One track timeline -> (values, class indices, mask).

    ``None`` label rows are absent keyframes and get mask=False with a
    zeroed feature row (padding). Values must already be causal.
    """
    if len(feature_rows) != len(label_rows):
        raise ValueError("feature and label rows must align")
    values: list[float] = []
    indices: list[int] = []
    mask: list[bool] = []
    for feat, label in zip(feature_rows, label_rows):
        if label is None:
            values.extend([0.0] * INPUT_DIM)
            indices.append(3)
            mask.append(False)
        else:
            if len(feat) != INPUT_DIM:
                raise ValueError("feature row must have 20 values")
            values.extend(feat)
            indices.append(class_index(label))
            mask.append(True)
    return values, indices, mask


def build_model(config: Mapping[str, Any]):
    """Construct the GRU classifier (requires torch)."""
    import torch
    from torch import nn

    class EvidenceGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=int(config["input_dim"]),
                hidden_size=int(config["hidden_dim"]),
                num_layers=int(config["num_layers"]),
                dropout=float(config["dropout"]),
                batch_first=True,
            )
            self.head = nn.Linear(
                int(config["hidden_dim"]), int(config["num_classes"])
            )

        def forward(self, x, h=None):  # x: [B, T, D]
            out, h = self.gru(x, h)
            return self.head(out), h  # per-step logits [B, T, C]

    return EvidenceGRU()


def balanced_class_weights(label_rows: Sequence[LabelClass | None]) -> list[float]:
    counts = [0, 0, 0, 0]
    for label in label_rows:
        if label is not None:
            counts[class_index(label)] += 1
    total = sum(counts)
    if total == 0:
        return [1.0, 1.0, 1.0, 1.0]
    return [total / (4.0 * c) if c else 1.0 for c in counts]


def evidence_loss(
    logits: Any,
    targets: Any,
    weights: Any,
    mask: Any,
    false_commit_weight: float,
    temporal_weight: float,
) -> Any:
    """CE over visible steps + false-commit penalty + temporal drift.

    ``logits`` [B, T, C], ``targets`` [B, T], ``mask`` [B, T] bool.
    False commit: predicting PERSISTENT_CHANGE (index 2) when the target
    is not PERSISTENT_CHANGE at a visible step. Temporal: L2 drift
    between consecutive step logits at visible steps.
    """
    import torch
    from torch import nn

    change_idx = class_index(LabelClass.PERSISTENT_CHANGE)
    per_step = nn.functional.cross_entropy(
        logits.permute(0, 2, 1), targets, weight=weights, reduction="none"
    )  # [B, T]
    per_step = per_step * mask.float()
    total = per_step.sum() / mask.float().sum().clamp(min=1.0)

    if false_commit_weight:
        wrong = (targets != change_idx).float() * mask.float()
        predicted_change = torch.argmax(logits, dim=2) == change_idx
        penalty = (predicted_change.float() * wrong).sum()
        total = total + false_commit_weight * penalty / mask.float().sum().clamp(min=1.0)

    if temporal_weight:
        drift = torch.nn.functional.mse_loss(
            logits[:, 1:, :], logits[:, :-1, :], reduction="none"
        ).sum(dim=2)  # [B, T-1]
        drift = drift * mask[:, 1:].float()
        total = total + temporal_weight * drift.sum() / mask[:, 1:].float().sum().clamp(min=1.0)
    return total


def predict_probabilities(
    model, x, mask, temperature: float = 1.0
) -> list[dict[LabelClass, float]]:
    """Per-step causal probabilities for one track (requires torch).

    x: [1, T, D] float tensor; mask: [T] bool tensor. The GRU is causal,
    so the logits at step k use only steps <= k.
    """
    import torch

    model.eval()
    with torch.no_grad():
        logits, _ = model(x)  # [1, T, C]
    out: list[dict[LabelClass, float]] = []
    for k in range(x.shape[1]):
        probs = torch.softmax(
            logits[0, k] / max(temperature, 1e-6), dim=0
        )
        values = {
            label: float(probs[idx])
            for label, idx in zip(CLASS_ORDER, range(4))
        }
        out.append(values if bool(mask[k]) else _uniform())
    return out


def _uniform() -> dict[LabelClass, float]:
    return {label: 0.25 for label in CLASS_ORDER}


def train_model(
    bundles: Sequence[TrainBundle],
    config: Mapping[str, Any],
    out_dir: str | Path,
    seed: int,
) -> dict[str, float]:
    """Full training loop (cluster-side; requires torch + GPU).

    Writes ``model.pt`` + ``train_summary.json`` into ``out_dir``.
    """
    import json

    import torch

    torch.manual_seed(seed)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    weights = torch.tensor(
        balanced_class_weights(_all_labels(bundles)), dtype=torch.float32
    ).to(next(model.parameters()).device)

    epochs = int(config["epochs"])
    patience = int(config["early_stop_patience"])
    batch = int(config["batch_size"])
    train_bundles = [b for b in bundles if b.split == "train"]
    val_bundles = [b for b in bundles if b.split == "val"] or train_bundles
    device = next(model.parameters()).device
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_val: float | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    def _epoch_loss(bundles: list[TrainBundle]) -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for bundle in bundles:
                x = bundle.features.to(device)
                y = bundle.labels.to(device)
                m = bundle.masks.to(device)
                for start in range(0, x.shape[0], batch):
                    xb, yb, mb = x[start : start + batch], y[start : start + batch], m[start : start + batch]
                    logits, _ = model(xb)
                    loss = evidence_loss(
                        logits,
                        yb,
                        weights,
                        mb,
                        float(config["false_commit_weight"]),
                        float(config["temporal_consistency_weight"]),
                    )
                    total += float(loss.detach()) * int(mb.sum())
                    count += int(mb.sum())
        return total / max(count, 1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bundle in train_bundles:
            x = bundle.features.to(device)
            y = bundle.labels.to(device)
            m = bundle.masks.to(device)
            perm = torch.randperm(x.shape[0])
            for start in range(0, x.shape[0], batch):
                idx = perm[start : start + batch]
                xb, yb, mb = x[idx], y[idx], m[idx]
                optimizer.zero_grad()
                logits, _ = model(xb)
                loss = evidence_loss(
                    logits,
                    yb,
                    weights,
                    mb,
                    float(config["false_commit_weight"]),
                    float(config["temporal_consistency_weight"]),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
        val_loss = _epoch_loss(val_bundles)
        history.append({"epoch": epoch, "train_loss": total_loss, "val_loss": val_loss})
        if best_val is None or val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), out_path / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    summary = {
        "seed": seed,
        "epochs_run": len(history),
        "feature_names": list(FEATURE_NAMES),
        "history_tail": history[-5:],
    }
    (Path(out_dir) / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return {"train_loss_last": history[-1]["train_loss"]}


def _all_labels(bundles: Sequence[TrainBundle]) -> list[LabelClass]:
    labels: list[LabelClass] = []
    for bundle in bundles:
        mask = bundle.masks
        lab = bundle.labels
        for t in range(mask.shape[1]):
            if bool(mask[0, t]):
                labels.append(CLASS_ORDER[int(lab[0, t])])
    return labels
