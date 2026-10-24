"""Causal feature extraction tests (no torch, no data files)."""

from __future__ import annotations

import math

from egopersist.contracts import Observation, TrackObservation
from egopersist.evidence.features import (
    FEATURE_NAMES,
    INPUT_DIM,
    TrackFeatureState,
)


def make_obs(
    t: float,
    pos=(1.0, 2.0, 3.0),
    disp: float = 0.0,
    rot: float = 0.0,
    co_vis: float = 1.0,
    depth: float = 0.9,
    area: int = 5000,
) -> Observation:
    return Observation(
        sequence_id="seq",
        frame_id=str(int(t * 2)),
        timestamp_s=t,
        instance_id="7",
        mask_area_px=area,
        depth_support=depth,
        co_visibility=co_vis,
        appearance_similarity=1.0,
        centroid_world_m=pos,
        extent_world_m=(0.2, 0.2, 0.3),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        displacement_m=disp,
        rotation_deg=rot,
    )


def view(t: float, committed=(1.0, 2.0, 3.0), **kwargs) -> TrackObservation:
    return TrackObservation(
        track_id="7",
        observation=make_obs(t, **kwargs),
        committed_position_m=committed,
    )


def test_feature_vector_shape_and_finiteness() -> None:
    state = TrackFeatureState()
    for i in range(10):
        values = state.update(view(i * 0.5, disp=0.02), dynamic=False)
        assert len(values) == INPUT_DIM == 20
        assert len(FEATURE_NAMES) == 20
        assert len(set(FEATURE_NAMES)) == 20
        assert all(math.isfinite(v) for v in values)


def test_first_frame_has_zero_history() -> None:
    state = TrackFeatureState()
    values = state.update(view(0.5), dynamic=False)
    assert values[FEATURE_NAMES.index("dt_since_last_s")] == 0.0
    assert values[FEATURE_NAMES.index("mask_area_delta_norm")] == 0.0
    assert values[FEATURE_NAMES.index("consecutive_visible")] == 0.0


def test_deviation_from_committed_is_causal() -> None:
    state = TrackFeatureState()
    committed = (1.0, 2.0, 3.0)
    near = state.update(view(0.5, pos=committed), dynamic=False)
    far = state.update(view(1.0, pos=(1.4, 2.0, 3.0)), dynamic=False)
    dev = FEATURE_NAMES.index("deviation_from_committed_m")
    assert far[dev] > 0.3
    assert near[dev] < 0.01


def test_ratio_saturates_above_threshold() -> None:
    state = TrackFeatureState()
    state.update(view(0.5), dynamic=False)
    values = state.update(
        view(1.0, pos=(2.0, 2.0, 3.0), disp=0.9), dynamic=False
    )
    assert values[FEATURE_NAMES.index("ratio_to_change_threshold")] == 1.0


def test_future_data_does_not_change_past_features() -> None:
    """Causality: replaying the prefix gives identical features."""
    prefix = [view(t, disp=0.01) for t in (0.5, 1.0, 1.5)]
    extended = prefix + [view(t, pos=(5.0, 5.0, 5.0), disp=0.9) for t in (2.0, 2.5)]

    state_a = TrackFeatureState()
    out_a = [state_a.update(v, False) for v in prefix]

    state_b = TrackFeatureState()
    out_b = [state_b.update(v, False) for v in extended][: len(prefix)]

    assert out_a == out_b


def test_visibility_windows_bounded() -> None:
    state = TrackFeatureState()
    for i in range(40):
        values = state.update(view(i * 0.5), dynamic=False)
        frac = values[FEATURE_NAMES.index("visible_fraction_16")]
        assert 0.0 <= frac <= 1.0
    assert values[FEATURE_NAMES.index("consecutive_visible")] <= 1.0
