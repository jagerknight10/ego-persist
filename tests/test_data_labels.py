"""Tests for the label-rule engine on constructed timelines."""

from __future__ import annotations

from pathlib import Path

import pytest

from egopersist.contracts import EventType, LabelClass
from egopersist.data import TrackSample, label_track, load_label_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "labels_v1.yaml"


def _config() -> "object":
    return load_label_config(CONFIG)


def _sample(t: float, pose: tuple[float, float, float], dynamic: bool = False):
    return TrackSample(
        timestamp_s=t,
        pose=pose,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        visible=True,
        dynamic=dynamic,
    )


def _hidden(t: float):
    return TrackSample(timestamp_s=t, pose=None, orientation_wxyz=None, visible=False, dynamic=False)


def test_stable_track_labels_stable() -> None:
    config = _config()
    samples = [_sample(t * 0.5, (1.0, 2.0, 3.0)) for t in range(20)]
    events, labels = label_track("s", "t", samples, config)
    assert all(label is LabelClass.STABLE for label in labels)
    assert any(event.event_type is EventType.STABLE for event in events)


def test_dynamic_track_labels_transient() -> None:
    config = _config()
    samples = [_sample(t * 0.5, (t * 0.1, 0.0, 0.0), dynamic=True) for t in range(20)]
    events, labels = label_track("s", "t", samples, config)
    assert all(label is LabelClass.TRANSIENT for label in labels)
    assert any(event.event_type is EventType.TRANSIENT for event in events)


def test_hidden_frames_are_unknown() -> None:
    config = _config()
    samples = [_sample(t * 0.5, (1.0, 2.0, 3.0)) for t in range(4)]
    samples += [_hidden(2.0 + t * 0.5) for t in range(4)]
    samples += [_sample(4.0 + t * 0.5, (1.0, 2.0, 3.0)) for t in range(4)]
    _, labels = label_track("s", "t", samples, config)
    assert labels[4:8] == [LabelClass.UNKNOWN] * 4


def test_durable_move_confirms_persistent_change() -> None:
    config = _config()
    samples = [_sample(t * 0.5, (1.0, 2.0, 3.0)) for t in range(4)]
    moved = [(1.5, 2.0, 3.0)] * 20  # 0.5 m jump, held for 10 s
    samples += [_sample(2.0 + t * 0.5, pose) for t, pose in enumerate(moved)]
    events, labels = label_track("s", "t", samples, config)
    change = [event for event in events if event.event_type is EventType.PERSISTENT_CHANGE]
    assert len(change) == 1
    assert labels.index(LabelClass.PERSISTENT_CHANGE) >= 4
    assert labels[-1] is LabelClass.PERSISTENT_CHANGE


def test_brief_displacement_reverts() -> None:
    config = _config()
    samples = [_sample(t * 0.5, (1.0, 2.0, 3.0)) for t in range(4)]
    samples.append(_sample(2.0, (1.2, 2.0, 3.0)))  # 0.2 m displacement
    samples.append(_sample(2.5, (1.0, 2.0, 3.0)))  # back within tolerance
    samples += [_sample(3.0 + t * 0.5, (1.0, 2.0, 3.0)) for t in range(8)]
    events, labels = label_track("s", "t", samples, config)
    reversion = [event for event in events if event.event_type is EventType.REVERSION]
    assert len(reversion) == 1
    assert labels[-1] is LabelClass.STABLE
    assert not any(event.event_type is EventType.PERSISTENT_CHANGE for event in events)


def test_move_under_threshold_stays_stable() -> None:
    config = _config()
    samples = [_sample(t * 0.5, (1.0, 2.0, 3.0)) for t in range(4)]
    samples += [_sample(2.0 + t * 0.5, (1.05, 2.0, 3.0)) for t in range(20)]
    events, labels = label_track("s", "t", samples, config)
    assert all(label is LabelClass.STABLE for label in labels)
    assert not any(event.event_type is EventType.PERSISTENT_CHANGE for event in events)


def test_labels_follow_threshold_changes(tmp_path: Path) -> None:
    """A 0.5 m held move is persistent at 0.10 m, stable at 0.75 m."""
    base = CONFIG.read_text().replace("translation_m: 0.10", "translation_m: 0.75")
    loose = tmp_path / "labels_loose.yaml"
    loose.write_text(base)
    config = load_label_config(CONFIG)
    loose_config = load_label_config(loose)
    samples = [_sample(t * 0.5, (1.0, 2.0, 3.0)) for t in range(4)]
    samples += [_sample(2.0 + t * 0.5, (1.5, 2.0, 3.0)) for t in range(20)]
    _, tight_labels = label_track("s", "t", samples, config)
    _, loose_labels = label_track("s", "t", samples, loose_config)
    assert LabelClass.PERSISTENT_CHANGE in tight_labels
    assert all(label is LabelClass.STABLE for label in loose_labels)


def test_causal_no_future_frames() -> None:
    """Extending the future never changes the past labels."""
    config = _config()
    prefix = [_sample(t * 0.5, (1.0, 2.0, 3.0)) for t in range(6)]
    mid = prefix + [_sample(3.0 + t * 0.5, (1.0, 2.0, 3.0)) for t in range(6)]
    _, labels_a = label_track("s", "t", prefix, config)
    _, labels_b = label_track("s", "t", mid, config)
    assert labels_b[: len(prefix)] == labels_a
