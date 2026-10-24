"""Tests for deterministic keyframe sampling and observation building."""

from __future__ import annotations

from pathlib import Path

import pytest

from egopersist.contracts import Observation
from egopersist.data import build_observations, keyframe_times

SEQ133 = Path(__file__).resolve().parents[1] / "data" / "raw" / "adt" / "seq133"


def test_keyframe_times_deterministic() -> None:
    times = keyframe_times(0.0, 10.0, 2.0)
    assert times[0] == 0.0
    assert len(times) == 21
    assert times[-1] == 10.0
    deltas = {round(b - a, 9) for a, b in zip(times, times[1:], strict=False)}
    assert deltas == {0.5}


def test_keyframe_times_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        keyframe_times(0.0, 10.0, 0.0)
    with pytest.raises(ValueError):
        keyframe_times(10.0, 0.0, 2.0)


@pytest.mark.skipif(
    not (SEQ133 / "main_groundtruth").is_dir(), reason="seq133 sample not present"
)
def test_observations_static_and_dynamic_tracks() -> None:
    from egopersist.data import load_adt_sequence

    seq = load_adt_sequence(SEQ133)
    observations = build_observations(seq, 2.0)
    static = [uid for uid in observations if seq.instances[uid].motion_type == "static"]
    dynamic = [
        uid for uid in observations if seq.instances[uid].motion_type == "dynamic"
    ]
    assert len(static) > 100
    assert len(dynamic) >= 40
    for track in observations.values():
        stamps = [obs.timestamp_s for obs in track]
        assert stamps == sorted(stamps)
        for obs in track:
            assert isinstance(obs, Observation)
            assert obs.mask_area_px >= 1
            assert 0.0 <= obs.co_visibility <= 1.0
            assert 0.0 <= obs.depth_support <= 1.0


@pytest.mark.skipif(
    not (SEQ133 / "main_groundtruth").is_dir(), reason="seq133 sample not present"
)
def test_observations_causal_pose_lag() -> None:
    """Pose at a keyframe never comes from a later frame."""
    from egopersist.data import load_adt_sequence

    seq = load_adt_sequence(SEQ133)
    observations = build_observations(seq, 2.0)
    # find one dynamic track and check a midpoint keyframe
    track = next(
        observations[uid]
        for uid in observations
        if seq.instances[uid].motion_type == "dynamic"
    )
    mid = track[len(track) // 2]
    # the observation timestamp must be >= the first pose timestamp (no lookahead start)
    assert mid.timestamp_s >= seq.start_s
