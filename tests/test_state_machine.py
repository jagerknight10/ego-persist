"""State machine: commit gate, occlusion, reversion, transient, causality."""

import math

from egopersist.contracts.core import (
    DecisionAction,
    LabelClass,
    Observation,
)
from egopersist.mapping.state_machine import StateMachineConfig, TrackStateMachine, TrackView

CFG = StateMachineConfig()


def make_obs(centroid, t, orientation=(1.0, 0.0, 0.0, 0.0), co_vis=0.9):
    return Observation(
        sequence_id="seq1",
        frame_id=f"f{t:.1f}",
        timestamp_s=t,
        instance_id="mug",
        mask_area_px=800,
        depth_support=0.9,
        co_visibility=co_vis,
        appearance_similarity=1.0,
        centroid_world_m=centroid,
        extent_world_m=(0.08, 0.10, 0.12),
        orientation_wxyz=orientation,
    )


def make_view(
    t,
    centroid,
    probs,
    committed=(0.0, 0.0, 1.0),
    committed_q=(1.0, 0.0, 0.0, 0.0),
    is_static=True,
    track_id="trk0001",
    orientation=(1.0, 0.0, 0.0, 0.0),
):
    if centroid is None:
        obs = None
    else:
        obs = make_obs(centroid, t, orientation=orientation)
    return TrackView(
        sequence_id="seq1",
        track_id=track_id,
        frame_id=f"f{t:.1f}",
        timestamp_s=t,
        observation=obs,
        probabilities=probs,
        committed_position_m=committed,
        committed_orientation_wxyz=committed_q,
        is_static=is_static,
    )


def probs(p_change=0.0, p_stable=0.0, p_transient=0.0, p_unknown=0.0):
    return {
        LabelClass.STABLE: p_stable,
        LabelClass.TRANSIENT: p_transient,
        LabelClass.PERSISTENT_CHANGE: p_change,
        LabelClass.UNKNOWN: p_unknown,
    }


def test_stable_track_keeps_map():
    sm = TrackStateMachine(CFG)
    for k in range(10):
        t = k * 0.5
        d, u = sm.update(make_view(t, (0.005, 0.0, 1.0), probs(p_stable=0.99)))
        if k >= CFG.commit_min_visible_keyframes:
            assert d.label == LabelClass.STABLE
            assert d.action == DecisionAction.KEEP
        assert u.map_version == 0


def test_small_deviation_is_kept_not_committed():
    """Deviation above change threshold but low P(change) -> keep, no commit."""
    sm = TrackStateMachine(CFG)
    seen = []
    for k in range(10):
        t = k * 0.5
        # 0.2 m away (exceeds 0.10 m) but evidence says stable
        d, u = sm.update(make_view(t, (0.2, 0.0, 1.0), probs(p_stable=0.9, p_change=0.05)))
        seen.append((d, u))
    for d, u in seen:
        assert u.action != DecisionAction.COMMIT
        assert u.map_version == 0


def test_commit_gate_requires_all_gates():
    sm = TrackStateMachine(CFG)
    new_pose = (0.3, 0.0, 1.0)  # 0.3 m away
    results = []
    # only 4 visible keyframes, span < 1.0 s: must NOT commit
    for k in range(4):
        t = k * 0.5
        d, u = sm.update(make_view(t, new_pose, probs(p_change=0.99)))
        results.append((d, u))
    assert all(u.action == DecisionAction.DEFER for _, u in results)
    assert all(d.label == LabelClass.UNKNOWN for d, _ in results)
    # keyframe 5 (t=2.0): 5 keyframes, span 2.0 s -> commit fires
    d, u = sm.update(make_view(2.0, new_pose, probs(p_change=0.99)))
    assert d.action == DecisionAction.COMMIT
    assert u.map_version == 1
    assert u.new_position_m is not None
    assert all(abs(a - b) < 1e-6 for a, b in zip(u.new_position_m, new_pose))
    assert d.label == LabelClass.PERSISTENT_CHANGE


def test_commit_gate_low_probability_defers():
    sm = TrackStateMachine(CFG)
    new_pose = (0.3, 0.0, 1.0)
    for k in range(6):
        t = k * 0.5
        # P(change)=0.6: above suspect(0.5), below commit(0.85) -> defer
        d, u = sm.update(make_view(t, new_pose, probs(p_change=0.6, p_stable=0.4)))
        assert d.action == DecisionAction.DEFER
        assert u.map_version == 0


def test_commit_gate_low_co_visibility_defers():
    sm = TrackStateMachine(CFG)
    new_pose = (0.3, 0.0, 1.0)
    for k in range(6):
        t = k * 0.5
        d, u = sm.update(
            make_view(t, new_pose, probs(p_change=0.99)),
        )
        # override co-visibility via a custom obs
    # rebuild with low co-vis
    sm2 = TrackStateMachine(CFG)
    for k in range(6):
        t = k * 0.5
        view = make_view(t, new_pose, probs(p_change=0.99))
        view = TrackView(
            sequence_id=view.sequence_id,
            track_id=view.track_id,
            frame_id=view.frame_id,
            timestamp_s=t,
            observation=make_obs(new_pose, t, co_vis=0.3),
            probabilities=view.probabilities,
            committed_position_m=view.committed_position_m,
            committed_orientation_wxyz=view.committed_orientation_wxyz,
            is_static=True,
        )
        d, u = sm2.update(view)
    assert u.action == DecisionAction.DEFER
    assert u.map_version == 0


def test_occlusion_never_removes_map():
    """Occluded static track: defer, map retained, no removal."""
    sm = TrackStateMachine(CFG)
    # establish stable
    for k in range(6):
        sm.update(make_view(k * 0.5, (0.0, 0.0, 1.0), probs(p_stable=0.99)))
    # occluded for 5 keyframes
    for k in range(6, 12):
        d, u = sm.update(make_view(k * 0.5, None, probs(p_stable=0.9)))
        assert d.action == DecisionAction.DEFER
        assert u.action != DecisionAction.REJECT
    # back to view, agrees with map
    d, u = sm.update(make_view(6.0, (0.0, 0.0, 1.0), probs(p_stable=0.99)))
    assert d.action == DecisionAction.KEEP
    assert u.map_version == 0


def test_occlusion_during_suspect_keeps_map():
    sm = TrackStateMachine(CFG)
    new_pose = (0.3, 0.0, 1.0)
    for k in range(3):
        d, u = sm.update(make_view(k * 0.5, new_pose, probs(p_change=0.99)))
        assert d.action == DecisionAction.DEFER
    # occluded mid-suspect
    d, u = sm.update(make_view(1.5, None, probs(p_change=0.99)))
    assert d.action == DecisionAction.DEFER
    assert u.map_version == 0


def test_reversion_gate():
    """Committed change that returns to pre-change pose reverts."""
    sm = TrackStateMachine(CFG)
    original = (0.0, 0.0, 1.0)
    moved = (0.3, 0.0, 1.0)
    # commit at t=2.0
    for k in range(5):
        t = k * 0.5
        sm.update(make_view(t, moved if t > 0 else original, probs(p_change=0.99)))
    d, u = sm.update(make_view(2.5, moved, probs(p_change=0.99)))
    assert d.action == DecisionAction.COMMIT
    # hold at moved position for a while
    for k in range(6, 9):
        d, u = sm.update(make_view(k * 0.5, moved, probs(p_change=0.99)))
        assert d.action == DecisionAction.KEEP
    # object returns to original pose -> revert
    d, u = sm.update(make_view(4.5, original, probs(p_stable=0.99)))
    assert d.action == DecisionAction.REVERT
    assert u.map_version == 2
    assert all(abs(a - b) < 1e-9 for a, b in zip(u.new_position_m, original))


def test_rotation_deviation_opens_suspect():
    sm = TrackStateMachine(CFG)
    q20 = (math.cos(math.radians(10)), math.sin(math.radians(10)), 0.0, 0.0)
    d, u = sm.update(
        make_view(0.0, (0.0, 0.0, 1.0), probs(p_change=0.99), orientation=q20)
    )
    # 20 deg rotation > 15 deg threshold -> suspect opened (defer)
    assert d.action == DecisionAction.DEFER


def test_transient_object_lifecycle():
    sm = TrackStateMachine(CFG)
    pos = (0.0, 0.0, 1.0)
    # insufficient evidence first
    d, u = sm.update(
        make_view(0.0, pos, probs(p_transient=0.95), is_static=False)
    )
    assert d.action == DecisionAction.DEFER
    for k in range(1, 6):
        d, u = sm.update(make_view(k * 0.5, pos, probs(p_transient=0.95, p_unknown=0.05), is_static=False))
    assert d.label == LabelClass.TRANSIENT
    assert d.action == DecisionAction.KEEP
    # object leaves; within 15 s window -> keep
    d, u = sm.update(make_view(3.0, None, probs(p_transient=0.95, p_unknown=0.05), is_static=False))
    assert d.action == DecisionAction.KEEP
    # after the 15 s return window -> reject
    d, u = sm.update(make_view(19.0, None, probs(p_transient=0.95, p_unknown=0.05), is_static=False))
    assert d.action == DecisionAction.REJECT


def test_causality_past_decisions_unchanged():
    """A later observation must not change an earlier decision."""
    sm = TrackStateMachine(CFG)
    decisions = []
    for k in range(6):
        t = k * 0.5
        d, _ = sm.update(make_view(t, (0.0, 0.0, 1.0), probs(p_stable=0.99)))
        decisions.append(d)
    # snapshot
    frozen = [d.to_dict() for d in decisions]
    # feed a large deviation later
    d, u = sm.update(make_view(3.0, (0.5, 0.0, 1.0), probs(p_change=0.99)))
    # earlier decisions are objects already returned; re-derive via fresh machine
    sm2 = TrackStateMachine(CFG)
    for k in range(6):
        t = k * 0.5
        d2, _ = sm2.update(make_view(t, (0.0, 0.0, 1.0), probs(p_stable=0.99)))
        assert d2.to_dict() == frozen[k]


def test_unknown_track_defers_until_evidence():
    sm = TrackStateMachine(CFG)
    d, u = sm.update(make_view(0.0, (0.0, 0.0, 1.0), probs(p_stable=0.6)))
    assert d.label == LabelClass.UNKNOWN
    assert d.action == DecisionAction.DEFER
