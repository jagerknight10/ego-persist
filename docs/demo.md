# Synthetic Demo

This demo is a deterministic CPU replay. It uses simple primitives and no
external data. It does not show ADT results.

## Run

```bash
PYTHONPATH=src python3.12 -m wise_egomap.cli.main generate \
  --output demo-output
```

Open `demo-output/index.html` in a browser. The file is self-contained. It
uses no network requests and no external assets.

Ask the mug query:

```bash
PYTHONPATH=src python3.12 -m wise_egomap.cli.main query \
  --scenario durable_mug_move --time 9.0
```

## Replay Schema

The schema identifier is `wise_egomap.demo.replay/1`.

Each scenario has:

- `scenario_id`, `title`, `summary`, and `duration_s`;
- `frames` in increasing time order;
- `key_events` for the timeline.

Each frame has:

- `timestamp_s` and `caption`;
- `observations` for the policy input;
- `ego_map` and `instant_map` for the comparison;
- `decisions` with labels, actions, confidence, probabilities, and reasons;
- `evidence` with values, units, thresholds, and bar ranges;
- `query` with the mug state, position, and confidence.

All records validate on load. Unknown fields cause a schema error. The replay
has a JSON round-trip and stable key order.

## Scenarios

| Scenario | Cause | Instant update | WISE-EgoMap |
|---|---|---|---|
| Passing person | Human candidate | Adds the person | Rejects the candidate |
| Durable mug move | New pose stays 5 s | Commits at once | Defers, then commits at 7.5 s |
| Occlusion | Mug mask absent | Removes the mug | Keeps the mug and defers |

The replay duration is 18.5 s in total. The viewer shows the WISE-EgoMap map,
the instant map, the map timeline, the decision reasons, the evidence bars,
and the mug query.

## State-Machine Adapter

`DemoStateMachineAdapter` is a small policy adapter. It applies the causal
commit gate and the non-observation rule. `InstantUpdateAdapter` is the
comparison policy. Both use the `StateMachineAdapter` protocol.

The future mapping state machine can replace the adapter with no change to the
replay schema or the viewer. The adapter receives only the current map and the
current keyframe observations.

## Video Render

Render a small local GIF:

```bash
PYTHONPATH=src python3.12 -m wise_egomap.cli.main render \
  --output demo-output/demo.gif --fps 2
```

Render one PNG frame:

```bash
PYTHONPATH=src python3.12 -m wise_egomap.cli.main render \
  --output demo-output/occlusion.png --scenario occlusion --frame 6
```

Render an MP4 on the cluster:

```bash
PYTHONPATH=src python3.12 -m wise_egomap.cli.main render \
  --output demo-output/demo.mp4 --fps 2
```

MP4 output needs FFmpeg. Use a saved replay with `--replay path/to/replay.json`
when the cluster must render a fixed artifact.

## Tests

```bash
PYTHONPATH=src python3.12 -m pytest -q
```

The tests cover schema round trip, deterministic replay, the three scenarios,
the query, the offline viewer, and the PNG render path.
