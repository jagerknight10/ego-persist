"""Sequence audit report.

The audit records what a run actually consumed: sequences, keyframes,
tracks, visibility, and label counts. Every field is required in the JSON
output so downstream stages can detect silent data loss.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from egopersist.contracts import EventLabel, LabelClass
from egopersist.data.keyframes import keyframe_times


def build_audit(
    sequence_id: str,
    *,
    n_objects: int,
    n_tracks: int,
    keyframe_times_s: list[float],
    label_config_path: str,
    observations_per_track: dict[str, int],
    events: list[EventLabel],
    visible_fraction: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the audit dictionary with a stable key order."""
    label_counts = Counter(event.label for event in events)
    event_counts = Counter(event.event_type.value for event in events)
    return {
        "schema": "egopersist.data.audit/1",
        "sequence_id": sequence_id,
        "n_objects": n_objects,
        "n_tracks": n_tracks,
        "n_keyframes": len(keyframe_times_s),
        "keyframe_start_s": keyframe_times_s[0] if keyframe_times_s else None,
        "keyframe_end_s": keyframe_times_s[-1] if keyframe_times_s else None,
        "n_observations": sum(observations_per_track.values()),
        "visible_fraction": round(visible_fraction, 6),
        "event_counts": dict(sorted(event_counts.items())),
        "label_counts": {
            label.value: int(label_counts.get(label, 0)) for label in LabelClass
        },
        "label_config_sha256": _sha256(Path(label_config_path)),
        "extra": dict(extra or {}),
    }


def write_audit(audit: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
