"""Create a small deterministic split file for cluster smoke tests."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path


def create_split(
    *,
    sequences: Sequence[str],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, object]:
    """Return a deterministic sequence-level split."""
    if any(not sequence or not sequence.strip() for sequence in sequences):
        raise ValueError("sequence names must be non-empty")
    unique = tuple(sorted(set(sequences)))
    if not unique:
        raise ValueError("sequences must not be empty")
    if len(unique) != len(tuple(sequences)):
        raise ValueError("sequences must be unique")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must not cover all sequences")
    if len(unique) < 3:
        raise ValueError("at least three sequences are necessary")

    ordered = list(unique)
    random.Random(seed).shuffle(ordered)
    train_count = max(1, int(len(ordered) * train_fraction))
    validation_count = max(1, int(len(ordered) * validation_fraction))
    if train_count + validation_count >= len(ordered):
        validation_count = len(ordered) - train_count - 1
    test_start = train_count + validation_count
    return {
        "version": 1,
        "seed": seed,
        "policy": "sequence_and_activity_instance",
        "train": ordered[:train_count],
        "validation": ordered[train_count:test_start],
        "test": ordered[test_start:],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write a split file with owner-only permissions."""
    arguments = parse_args(argv)
    split = create_split(
        sequences=arguments.sequence,
        seed=arguments.seed,
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arguments.output.chmod(0o600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
