"""Split definitions and validation.

A split is a named list of sequence IDs. Sequence-level splitting is the
minimum unit: one sequence never appears in two splits. Object-level overlap
within a sequence is handled by the association stage, not by splits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitSpec:
    """Named sequence split."""

    name: str
    sequence_ids: tuple[str, ...]


#: Aliases accepted in split files. ``validation`` is canonicalised to
#: ``val`` so cluster split writers and stage readers agree on one name.
SPLIT_ALIASES = {"validation": "val", "test": "test"}


def normalize_split_raw(raw: dict[str, list[str]]) -> dict[str, list[str]]:
    """Apply split-name aliases and reject collisions."""
    out: dict[str, list[str]] = {}
    for name, items in raw.items():
        canonical = SPLIT_ALIASES.get(name, name)
        if canonical in out:
            raise ValueError(
                f"duplicate split: {name!r} and {canonical!r} alias to one name"
            )
        out[canonical] = list(items)
    return out


def load_splits(path: str | Path) -> dict[str, SplitSpec]:
    """Load a split manifest: ``{"train": [...], "val": [...], ...}``.

    ``validation`` is accepted as an alias for ``val``.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError("split manifest must be a non-empty object")
    splits: dict[str, SplitSpec] = {}
    for name, items in normalize_split_raw(raw).items():
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item for item in items
        ):
            raise ValueError(f"split {name!r} must be a list of sequence ids")
        if len(set(items)) != len(items):
            raise ValueError(f"split {name!r} contains duplicate sequence ids")
        splits[name] = SplitSpec(name=name, sequence_ids=tuple(items))
    return splits


def validate_splits(
    splits: dict[str, SplitSpec], available: tuple[str, ...] | set[str]
) -> None:
    """Reject overlaps and unknown sequence IDs."""
    available = set(available)
    seen: dict[str, str] = {}
    for name in sorted(splits):
        for sequence_id in splits[name].sequence_ids:
            if sequence_id not in available:
                raise ValueError(
                    f"split {name!r} references unknown sequence {sequence_id!r}"
                )
            if sequence_id in seen:
                raise ValueError(
                    f"sequence {sequence_id!r} is in both "
                    f"{seen[sequence_id]!r} and {name!r}"
                )
            seen[sequence_id] = name
