# EgoPersist

EgoPersist is a causal map-maintenance layer for egocentric video.

It decides whether a map discrepancy is transient, deferred, or a durable
change. It does not train a new SLAM model. It sits above a frozen pose, depth,
segmentation, and appearance stack.

> Status: v0.1.0. All modules merged and tested. Training and reported
> results run on the university compute cluster (see docs/CLUSTER_RUNBOOK.md).
> This repository does not contain licensed data.

## Core Decision

| State | Meaning | Map action |
|---|---|---|
| `STABLE` | Observation agrees with the map | Keep |
| `TRANSIENT` | Discrepancy is short-lived or from a passing entity | Reject |
| `PERSISTENT_CHANGE` | New state passes the evidence gate | Commit |
| `UNKNOWN` | Low visibility or likely occlusion | Defer |

## Local Demo

```bash
PYTHONPATH=src python3.12 -m egopersist.cli.main query
PYTHONPATH=src python3.12 -m egopersist.cli.main generate --output demo-out
open demo-out/index.html
```

The demo is a scripted synthetic replay. It is not an ADT result.

## Local Checks


```bash
PYTHONPATH=src python3.12 -m pytest
```

## Repository Layout

```text
configs/              Versioned label and experiment settings
docs/                 Plan, protocol, cluster runbook, and agent briefs
src/egopersist/
  contracts/          Stable typed records
  data/               ADT adapter and label tools
  geometry/           Projection and 3D operations
  association/        Track association
  evidence/           Evidence features, Bayes, and GRU
  mapping/            Map state machine
  evaluation/         Metrics and event reports
  visualization/      Replay and query demo
jobs/                 Slurm templates
scripts/              Cluster entry points
tests/                Unit and integration tests
```

## Data Rules

- Do not commit ADT data, checkpoints, or download manifests.
- Do not download data on a laptop.
- Run data preparation and training on the university cluster.
- Keep raw data outside the repository.
- Keep fixtures small and synthetic.

## Documentation

- [Build plan](docs/PLAN.md)
- [Task board](TASKS.md)
- [Protocol](docs/PROTOCOL.md)
- [Cluster runbook](docs/CLUSTER_RUNBOOK.md)

## Licence

Code licence is MIT. Dataset and model licences remain separate.
