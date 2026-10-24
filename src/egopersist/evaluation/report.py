"""Machine-readable evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from egopersist.contracts import Decision, EventLabel, MapUpdate

from egopersist.evaluation.matching import EventMatching, match_events
from egopersist.evaluation.metrics import (
    commit_timing,
    final_position_error_m,
    persistent_change_metrics,
    persistent_change_metrics_by_sequence,
    transient_false_commit_rate,
    unknown_false_removal_rate,
    update_utility_at_0_30_m,
)
from egopersist.evaluation.models import (
    ActionEvent,
    PositionReference,
    empty_case_policy,
)
from egopersist.evaluation.pareto import ParetoPoint, pareto_plot_data
from egopersist.evaluation.taxonomy import classify_errors

SCHEMA_VERSION = "evaluation_v1"


def evaluation_report(
    events: Iterable[EventLabel],
    actions: Iterable[Decision | MapUpdate | ActionEvent],
    *,
    updates: Iterable[MapUpdate] = (),
    references: Iterable[PositionReference] = (),
    system_id: str = "system",
) -> dict[str, Any]:
    """Build a stable evaluation report.

    Pooled metrics use summed counts. Per-sequence metrics are separate. The
    report never computes a mean of ratios.
    """
    event_list = tuple(events)
    action_list = tuple(actions)
    update_list = tuple(updates)
    reference_list = tuple(references)
    matching: EventMatching = match_events(event_list, action_list)
    timing = commit_timing(event_list, action_list)
    position_error = final_position_error_m(update_list, reference_list)
    utility = update_utility_at_0_30_m(update_list, reference_list)
    false_commits = transient_false_commit_rate(event_list, action_list)
    stale_mean = timing.mean_stale_map_duration_s
    point = ParetoPoint(
        system_id=system_id,
        transient_false_commit_rate=false_commits.rate,
        stale_map_duration_s=stale_mean,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "system_id": system_id,
        "empty_case_policy": empty_case_policy(),
        "aggregation": {
            "micro": "pooled event and action counts",
            "macro": "separate per-sequence records",
            "mean_of_ratios_computed": False,
        },
        "event_matching": {
            "matched_count": len(matching.matches),
            "unmatched_event_count": len(matching.unmatched_events),
            "unmatched_action_count": len(matching.unmatched_actions),
            "details": matching.to_dict(),
        },
        "persistent_change_micro": persistent_change_metrics(
            event_list, action_list, matching=matching
        ).to_dict(),
        "persistent_change_by_sequence": [
            record.to_dict()
            for record in persistent_change_metrics_by_sequence(
                event_list, action_list
            )
        ],
        "transient_false_commit": false_commits.to_dict(),
        "unknown_false_removal": unknown_false_removal_rate(
            event_list, action_list
        ).to_dict(),
        "commit_timing": timing.to_dict(),
        "final_position_error": position_error.to_dict(),
        "update_utility_at_0_30_m": utility.to_dict(),
        "error_taxonomy": classify_errors(
            event_list, action_list, matching=matching
        ).to_dict(),
        "pareto_point": point.to_dict(),
    }
    return report


def pareto_report(points: Iterable[ParetoPoint]) -> dict[str, Any]:
    """Build a machine-readable Pareto report."""
    data = pareto_plot_data(points)
    return {
        "schema_version": SCHEMA_VERSION,
        "empty_case_policy": empty_case_policy(),
        "pareto": data,
    }


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a deterministic JSON report and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, sort_keys=True, indent=2, separators=(",", ": "))
    output.write_text(f"{text}\n", encoding="utf-8")
    return output
