"""Causal evidence model: features, Bayes accumulator, GRU, calibration."""

from egopersist.evidence.baselines import baseline_predict, majority_label
from egopersist.evidence.features import (
    FEATURE_NAMES,
    INPUT_DIM,
    TrackFeatureState,
)

__all__ = [
    "FEATURE_NAMES",
    "INPUT_DIM",
    "TrackFeatureState",
    "baseline_predict",
    "majority_label",
]
