"""Temperature scaling fitted on the validation split only.

Pure math (no torch required): the caller passes logits for each
validation keyframe; the fitter searches a one-parameter temperature on a
grid and keeps the value that minimises NLL on val. Training data must
never influence the temperature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def nll(logits: Sequence[float], target: int) -> float:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    p = exps[target] / total
    return -math.log(max(p, 1e-12))


def fit_temperature(
    val_logits: list[list[float]],
    val_targets: list[int],
    grid: Sequence[float] = tuple(
        round(0.25 + 0.25 * i, 2) for i in range(13)  # 0.25 .. 3.5
    ),
) -> float:
    """Grid-search temperature minimising mean NLL on val."""
    if not val_logits or len(val_logits) != len(val_targets):
        raise ValueError("need aligned non-empty val logits and targets")
    best_t = 1.0
    best_nll = math.inf
    for t in grid:
        mean = sum(nll(logits, target) for logits, target in zip(val_logits, val_targets)) / len(val_logits)
        if mean < best_nll:
            best_nll = mean
            best_t = t
    return best_t


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    val_nll: float

    def scaled(self, logits: Sequence[float]) -> list[float]:
        return [x / self.temperature for x in logits]
