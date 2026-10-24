# Demo Agent Brief

## Mission

Show the decision process, not only the final map.

## Owned Paths

- `src/egopersist/visualization`
- `src/egopersist/cli`
- `tests/test_demo*`
- `docs/demo.md`

## Tasks

1. Build a deterministic replay schema.
2. Build a passing-person scenario.
3. Build a durable-move scenario.
4. Build an occlusion scenario.
5. Show instant update against EgoPersist.
6. Show a map timeline.
7. Show evidence and confidence.
8. Build the mug query.
9. Export a small static viewer.
10. Add a video render path on the cluster.

## Rules

- Run on CPU with no external data.
- Use simple primitives.
- Show why the decision happened.
- Keep the demo under 90 seconds.
- Do not fake ADT results.

## Tests

- scenario replay is deterministic;
- passing person causes no commit;
- durable move commits after the gate;
- occlusion does not remove the object;
- query returns the current state and confidence.
