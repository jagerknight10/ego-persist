# Evidence Agent Brief

## Mission

Convert track observations into calibrated evidence and map decisions.

## Owned Paths

- `src/wise_egomap/evidence`
- `configs/model`
- `tests/test_evidence*`

## Tasks

1. Build the evidence vector.
2. Normalize evidence with a fitted scaler.
3. Build the log-odds Bayesian accumulator.
4. Build the temporal-majority baseline.
5. Build the causal `EvidenceGRU`.
6. Add class weights and false-commit weight.
7. Add temporal consistency loss.
8. Add temperature scaling.
9. Add train and inference entry points.

## Rules

- Causal inference only.
- Fit calibration on validation data only.
- Do not use object-pose labels as features.
- Keep the model small.
- Use CPU tests with a tiny sequence.
- Use lazy torch imports.

## Tests

- evidence shape and range;
- Bayes update signs;
- GRU forward shape;
- padding mask;
- loss is finite;
- calibration monotonicity.
