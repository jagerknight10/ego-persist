"""Training-free baselines for the four-label track classifier.

Baselines share the exact causal feature pipeline of the learned model,
so every comparison is apples-to-apples on identical inputs.
"""

from __future__ import annotations

from wise_egomap.contracts import LabelClass
from wise_egomap.evidence.features import INPUT_DIM


def majority_label(
    history: list[LabelClass], current: LabelClass, min_history: int = 3
) -> LabelClass:
    """Temporal majority over the last ``min_history`` labels.

    Ties resolve to ``UNKNOWN``; the causal contract is that the output
    at keyframe k uses only labels from keyframes < k.
    """
    window = history[-min_history:]
    if len(window) < min_history:
        return current
    counts = {label: window.count(label) for label in set(window)}
    best = max(counts.values())
    winners = [label for label, n in counts.items() if n == best]
    if LabelClass.UNKNOWN in winners or len(winners) > 1:
        return LabelClass.UNKNOWN
    return winners[0]


def baseline_predict(values: tuple[float, ...], dynamic: bool) -> LabelClass:
    """Rule baseline over one 20-d causal feature vector.

    Returns the most supported label for this keyframe alone (no window).
    """
    if len(values) != INPUT_DIM:
        raise ValueError("expected one feature vector of length 20")
    if dynamic:
        return LabelClass.TRANSIENT
    if values[9] >= 1.0:  # ratio_to_change_threshold saturated
        return LabelClass.PERSISTENT_CHANGE
    if values[13] < 0.6 or values[12] < 0.35:
        return LabelClass.UNKNOWN
    return LabelClass.STABLE
