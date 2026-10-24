# Evaluation Agent Brief

## Mission

Measure map-maintenance quality at event level and frame level.

## Owned Paths

- `src/egopersist/evaluation`
- `tests/test_evaluation*`

## Tasks

1. Implement event matching.
2. Implement precision, recall, and F1.
3. Implement transient false-commit rate.
4. Implement unknown false-removal rate.
5. Implement time to correct commit.
6. Implement stale-map duration.
7. Implement final position error.
8. Implement update utility.
9. Implement Pareto plotting data.
10. Implement error taxonomy.

## Rules

- Metrics have explicit units.
- Empty cases return documented values.
- Macro and micro results are separate.
- Do not average ratios.
- Write machine-readable reports.

## Tests

- known small example;
- perfect prediction;
- never-update case;
- instant-update case;
- missing event case;
- deterministic ordering.
