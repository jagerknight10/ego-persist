# Data Agent Brief

## Mission

Make ADT data readable, auditable, and labelable without copying data.

## Owned Paths

- `src/wise_egomap/data`
- `configs/data`
- `configs/labels_v1.yaml`
- `tests/test_data*`
- `tests/fixtures`
- `docs/data_card.md`

## Tasks

1. Build a small synthetic ADT-like fixture.
2. Build a backend protocol for ADT readers.
3. Read camera calibration, poses, depth, masks, and object tracks.
4. Build keyframe sampling.
5. Build the label-rule engine.
6. Build the sequence audit tool.
7. Build split validation.
8. Emit a data card.

## Rules

- Ground-truth object pose is never an input feature.
- Store sequence IDs, not raw data, in Git.
- Use explicit time units in field names.
- Use explicit length units in field names.
- Use a stable JSON schema for audit output.

## Tests

- fixture loads;
- keyframe selection is deterministic;
- labels follow threshold changes;
- split overlap is rejected;
- audit report has required fields.
