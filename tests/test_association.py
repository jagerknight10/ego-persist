"""Gated Hungarian association: matching, gating, and track lifecycle."""

from wise_egomap.association.association import TrackBank, default_gate
from wise_egomap.contracts.core import Observation


def make_obs(
    i: int,
    centroid,
    t: float,
    extent=(0.1, 0.1, 0.1),
    orientation=(1.0, 0.0, 0.0, 0.0),
) -> Observation:
    return Observation(
        sequence_id="seq1",
        frame_id=f"f{t:.1f}-{i}",
        timestamp_s=t,
        instance_id=f"obj{i}",
        mask_area_px=500,
        depth_support=0.9,
        co_visibility=0.9,
        appearance_similarity=1.0,
        centroid_world_m=centroid,
        extent_world_m=extent,
        orientation_wxyz=orientation,
    )


def test_gate_formula():
    assert default_gate(0.0) == 0.3
    assert default_gate(0.1) == 0.3
    assert abs(default_gate(0.5) - 1.0) < 1e-12


def test_single_match_and_new_track():
    bank = TrackBank(confirm_hits=2)
    obs0 = make_obs(0, (0.0, 0.0, 2.0), 0.0)
    res = bank.associate([obs0], 0.3)
    assert res.matches == []
    assert res.unmatched_obs == [0]
    [t0] = bank.apply(res, [obs0])
    assert t0.track_id == "trk0001"
    assert not t0.confirmed

    # next keyframe: same object seen again -> matched, now confirmed
    obs1 = make_obs(0, (0.01, 0.0, 2.0), 0.5)
    res = bank.associate([obs1], 0.3)
    assert res.matches == [(0, 0)]
    assert res.unmatched_obs == []
    bank.apply(res, [obs1])
    assert t0.confirmed
    assert t0.hits == 2


def test_gate_rejects_far_observation():
    bank = TrackBank(confirm_hits=2)
    bank.apply(
        bank.associate([make_obs(0, (0.0, 0.0, 2.0), 0.0)], 0.3),
        [make_obs(0, (0.0, 0.0, 2.0), 0.0)],
    )
    # observation 1 m away: must not match, must start a new track
    far = make_obs(1, (1.0, 0.0, 2.0), 0.5)
    res = bank.associate([far], 0.3)
    assert res.matches == []
    assert res.unmatched_obs == [0]
    [t1] = bank.apply(res, [far])
    assert t1.track_id == "trk0002"
    assert len(bank.tracks) == 2


def test_two_to_two_assignment():
    bank = TrackBank(confirm_hits=1)
    o0a = make_obs(0, (0.0, 0.0, 2.0), 0.0)
    o0b = make_obs(1, (1.0, 0.0, 2.0), 0.0)
    bank.apply(bank.associate([o0a, o0b], 0.3), [o0a, o0b])
    assert len(bank.tracks) == 2
    # swap-ish update: each obs stays with its own track (closest)
    o1a = make_obs(0, (0.0, 0.05, 2.0), 0.5)
    o1b = make_obs(1, (1.0, -0.05, 2.0), 0.5)
    res = bank.associate([o1a, o1b], 0.3)
    assert sorted(res.matches) == [(0, 0), (1, 1)]


def test_missing_tracks_detected():
    bank = TrackBank(confirm_hits=1)
    bank.apply(
        bank.associate([make_obs(0, (0.0, 0.0, 2.0), 0.0)], 0.3),
        [make_obs(0, (0.0, 0.0, 2.0), 0.0)],
    )
    # at its own keyframe time it is not missing
    assert bank.missing_tracks(0.0) == []
    # at the next keyframe, without a match, it is missing
    assert len(bank.missing_tracks(0.5)) == 1


def test_empty_bank_all_unmatched():
    bank = TrackBank(confirm_hits=3)
    res = bank.associate([make_obs(0, (0.0, 0.0, 2.0), 0.0)], 0.3)
    assert res.matches == []
    assert res.unmatched_obs == [0]
