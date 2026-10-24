# Evaluation Metrics

This package writes deterministic JSON reports. Distances use metres. Times
and durations use seconds. Ground truth is an evaluation reference only. It
is not a model input.

## Matching

An event matches one action if these conditions are true:

1. The sequence and track agree.
2. The action time is in the closed event window.
3. The action equals the expected action for the event type.

Each event and each action can have at most one match. Matching uses sorted
inputs to give the same result for the same data.

## Scores

- Precision, recall, and F1 use pooled counts. The positive class is
  `PERSISTENT_CHANGE`.
- Per-sequence metrics are separate records. The report does not average
  ratios.
- Transient false-commit rate is the number of transient events with a commit
  divided by the number of transient events.
- Unknown false-removal rate is the number of unknown or occlusion events
  with a reject divided by the number of those events.
- Time to correct commit is the first commit time minus the event start time.
- Stale-map duration is the delay to the first commit. If no commit occurs, it
  is the full event duration.
- Final position error uses the last map update at or before the last
  reference time.
- Update utility at 0.30 m is +1 for a useful commit, -1 for a harmful commit,
  and 0 for a neutral commit.
- Pareto data minimizes transient false-commit rate and stale-map duration.

## Empty Cases

The report includes `empty_case_policy`. These values also apply to the
standalone functions:

| Case | Value |
|---|---|
| No predicted positives: precision | 0.0 |
| No actual positives: recall | 1.0 |
| Undefined F1 | 0.0 |
| Zero rate denominator | 0.0 |
| No commit timing values: mean | `null` |
| No stale events: total duration | 0.0 s |
| No stale events: mean duration | 0.0 s |
| No position errors: mean | `null` |
| No update utility values: mean | 0.0 |
| No taxonomy items | 0 |
| No Pareto points | 0 |
