"""Bayes accumulator tests (no torch)."""

from __future__ import annotations

import math

from egopersist.contracts import LabelClass
from egopersist.evidence.bayes import BayesAccumulator
from egopersist.evidence.features import TrackFeatureState
from tests.test_evidence_features import view


def test_probabilities_sum_to_one() -> None:
    acc = BayesAccumulator(window=4)
    state = TrackFeatureState()
    for i in range(6):
        state.update(view(i * 0.5), dynamic=False)
    probs = acc.probabilities()
    assert set(probs) == set(LabelClass)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_window_is_bounded_fifo() -> None:
    from egopersist.evidence.features import INPUT_DIM

    acc = BayesAccumulator(window=2)
    for _ in range(4):
        acc.observe(tuple([0.0] * INPUT_DIM), False)
    assert len(acc._queue) == 2
    probs = acc.probabilities()
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_empty_accumulator_is_uniform() -> None:
    acc = BayesAccumulator(window=5)
    probs = acc.probabilities()
    assert all(abs(p - 0.25) < 1e-9 for p in probs.values())


def test_static_track_prefers_stable() -> None:
    acc = BayesAccumulator(window=5)
    state = TrackFeatureState()
    for i in range(8):
        values = state.update(view(i * 0.5), dynamic=False)
        acc.observe(values, False)
    assert acc.probabilities()[LabelClass.STABLE] > 0.5


def test_deviation_prefers_persistent_change() -> None:
    acc = BayesAccumulator(window=5)
    state = TrackFeatureState()
    for i in range(8):
        values = state.update(
            view(i * 0.5, pos=(1.5, 2.0, 3.0)), dynamic=False
        )
        acc.observe(values, False)
    probs = acc.probabilities()
    assert probs[LabelClass.PERSISTENT_CHANGE] > probs[LabelClass.STABLE]


def test_dynamic_fast_track_prefers_transient() -> None:
    acc = BayesAccumulator(window=5)
    state = TrackFeatureState()
    for i in range(8):
        values = state.update(
            view(i * 0.5, pos=(i * 0.3, 0.0, 1.0), disp=0.3, rot=5.0),
            dynamic=True,
        )
        acc.observe(values, True)
    assert acc.probabilities()[LabelClass.TRANSIENT] > 0.5
