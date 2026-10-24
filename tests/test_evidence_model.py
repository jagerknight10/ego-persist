"""Evidence GRU tests. torch-dependent paths are skipped without torch."""

from __future__ import annotations

from pathlib import Path

import pytest

from egopersist.contracts import LabelClass
from egopersist.evidence import baselines, calibrate
from egopersist.evidence.features import INPUT_DIM
from egopersist.evidence.model import (
    CLASS_ORDER,
    balanced_class_weights,
    class_index,
    load_model_config,
    pack_track,
)

def test_class_index_covers_all_labels() -> None:
    assert [class_index(label) for label in CLASS_ORDER] == [0, 1, 2, 3]
    assert len(CLASS_ORDER) == 4


def test_pack_track_aligns_and_masks() -> None:
    rows = [[0.1] * INPUT_DIM, [0.2] * INPUT_DIM, [0.3] * INPUT_DIM]
    labels = [LabelClass.STABLE, None, LabelClass.TRANSIENT]
    values, indices, mask = pack_track(rows, labels)
    assert len(values) == INPUT_DIM * 3
    assert mask == [True, False, True]
    assert indices == [0, 3, 1]  # padded row takes UNKNOWN
    assert values[INPUT_DIM] == 0.0  # padded row is zeroed


def test_pack_track_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError):
        pack_track([[0.1] * INPUT_DIM], [])


def test_balanced_weights_inverse_to_counts() -> None:
    labels = [
        LabelClass.STABLE,
        LabelClass.STABLE,
        LabelClass.STABLE,
        LabelClass.STABLE,
        LabelClass.TRANSIENT,
    ]
    weights = balanced_class_weights(labels)
    assert len(weights) == 4
    assert weights[0] < weights[3]  # rare class weighted higher
    assert weights[1] > weights[0]


def test_balanced_weights_empty() -> None:
    assert balanced_class_weights([]) == [1.0, 1.0, 1.0, 1.0]


def test_load_model_config(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "model" / "evidence_gru.yaml"
    config = load_model_config(config_path)
    assert config["input_dim"] == 20
    assert config["num_classes"] == 4
    assert config["hidden_dim"] == 128


def test_calibration_fit_and_scale() -> None:
    logits = [[5.0, 1.0, 1.0, 1.0], [1.0, 5.0, 1.0, 1.0]]
    targets = [0, 1]
    t = calibrate.fit_temperature(logits, targets)
    assert 0.25 <= t <= 3.5
    result = calibrate.CalibrationResult(temperature=t, val_nll=0.1)
    logits_full = [2.0, 1.0, 0.5, 0.0]
    scaled = result.scaled(logits_full)
    # argmax invariant under positive temperature scaling
    assert max(range(4), key=lambda i: logits_full[i]) == max(
        range(4), key=lambda i: scaled[i]
    )


def _values(**overrides):
    base = {
        "visible": 1.0,
        "co_visibility": 0.9,
        "depth_support": 0.8,
        "ratio_to_change_threshold": 0.2,
    }
    base.update(overrides)
    from egopersist.evidence.features import FEATURE_NAMES

    out = [0.0] * INPUT_DIM
    for name, value in base.items():
        out[FEATURE_NAMES.index(name)] = value
    return tuple(out)


def test_baselines_deterministic() -> None:
    assert baselines.baseline_predict(_values(), False) is LabelClass.STABLE
    assert baselines.baseline_predict(_values(), True) is LabelClass.TRANSIENT
    assert (
        baselines.baseline_predict(
            _values(ratio_to_change_threshold=1.0), False
        )
        is LabelClass.PERSISTENT_CHANGE
    )
    assert (
        baselines.baseline_predict(_values(co_visibility=0.2), False)
        is LabelClass.UNKNOWN
    )
    history = [LabelClass.STABLE, LabelClass.STABLE, LabelClass.STABLE]
    assert (
        baselines.majority_label(history, LabelClass.UNKNOWN)
        is LabelClass.STABLE
    )
    assert (
        baselines.majority_label(history[:2], LabelClass.STABLE)
        is LabelClass.STABLE
    )  # below min_history -> current


def test_gru_forward_and_loss_finite() -> None:
    torch = pytest.importorskip("torch")
    from egopersist.evidence.model import build_model, evidence_loss, load_model_config

    config = load_model_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "model"
        / "evidence_gru.yaml"
    )
    model = build_model(config)
    x = torch.randn(4, 6, INPUT_DIM)
    logits, h = model(x)
    assert logits.shape == (4, 6, 4)
    y = torch.tensor([[0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 2],
                      [2, 2, 2, 2, 2, 3], [3, 3, 3, 3, 3, 0]])
    m = torch.ones(4, 6, dtype=torch.bool)
    m[0, -1] = False
    w = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss = evidence_loss(logits, y, w, m, 3.0, 0.1)
    value = float(loss.detach())
    assert value == value and value >= 0.0  # finite

