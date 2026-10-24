# WISE-EgoMap One-Week Build Plan

Status: ACCEPTED. Defaults are confirmed. The build order is now:

1. synthetic demo;
2. full code and tests;
3. cluster package;
4. ADT work after licence approval.

## 1. Goal

Build a tested, open repository for causal map maintenance on egocentric video.

The system must decide one state for each 3D instance track:

- `STABLE`
- `TRANSIENT`
- `PERSISTENT_CHANGE`
- `UNKNOWN`

The repository must include:

- a reproducible ADT data protocol;
- label generation and audit tools;
- geometry and track association code;
- a Bayesian baseline;
- an `EvidenceGRU` model;
- a map state machine;
- baselines, metrics, and Pareto evaluation;
- a synthetic local demo;
- a viewer and a query flow;
- cluster jobs and exact run instructions.

## 2. Scope Rules

1. ADT is the only quantitative dataset in this build.
2. Lost & Found is a qualitative comparison only.
3. HOI4D and Argos are out of scope for the one-week build.
4. The first protocol uses ground-truth masks, depth, and poses.
5. The second protocol uses frozen SAM 2 masks and DINOv2 features.
6. MPS or DROID-W is an extension, not a blocker.
7. No foundation model is trained from scratch.
8. No dataset or file over 100 MB is downloaded on this machine.
9. The repository contains only code, configs, docs, and tiny fixtures.
10. Reported ADT results run on the university cluster.

## 3. Day Plan

| Day | Work | Gate |
|---|---|---|
| 0 | Lock scope, contracts, repo layout, and owner map | Defaults accepted |
| 1 | Scaffold, configs, CI, fixtures, geometry core | CPU tests pass |
| 2 | ADT adapter, label builder, audit tool, association | Synthetic event chain runs |
| 3 | Evidence features, Bayes model, EvidenceGRU, state machine | Training smoke test passes |
| 4 | Baselines, metrics, Pareto report, error taxonomy | Fixture results match expected values |
| 5 | Viewer, query flow, demo replay, cluster jobs | Local demo runs without real data |
| 6 | Integration, docs, container, release checks | CI is clean |
| 7 | Cluster dry run and fixes | One sample sequence and one train job complete |

## 4. Agent Split

Each agent uses a separate Git worktree. The lead owns shared interfaces and merges.

| Agent | Owns | Must deliver |
|---|---|---|
| Lead | Contracts, integration, release | Interface freeze, review, release candidate |
| Data | `wise_egomap/data`, ADT scripts, splits | ADT adapter, label rules, audit report, toy fixture |
| Geometry | `wise_egomap/geometry`, `association`, `mapping` | Projection, tracks, state machine, update logic |
| Evidence | `wise_egomap/evidence`, baselines, training | Features, Bayes model, EvidenceGRU, calibration |
| Evaluation | `wise_egomap/evaluation`, reports | Metrics, event evaluator, Pareto plots, error taxonomy |
| Demo | `wise_egomap/visualization`, demo app | Replay, viewer, query flow, video scaffold |
| Infra | `environment`, `jobs`, CI | Container, Slurm jobs, provenance, release checks |

## 5. Shared Contracts

Freeze these before parallel work starts:

1. `TrackObservation`
2. `EvidenceVector`
3. `Decision`
4. `MapUpdate`
5. `EventLabel`
6. `RunManifest`

All contracts use typed Python dataclasses or Pydantic models. All contracts have JSON round-trip tests.

## 6. Merge Order

1. Lead: repository scaffold and contracts.
2. Data: toy fixture and ADT adapter.
3. Geometry: projection, association, and state machine.
4. Evidence: baselines and model.
5. Evaluation: metrics and reports.
6. Demo: viewer and query flow.
7. Infra: container, jobs, and release.
8. Lead: integration, docs, and tag.

## 7. Acceptance Criteria

### Local

- `pytest` passes on CPU.
- `ruff check` passes.
- `ruff format --check` passes.
- The synthetic demo replays three cases:
  - passing person;
  - durable object move;
  - occlusion.
- The demo shows `defer`, `reject`, and `commit` decisions.
- The query flow answers "Where is the mug?" with state, position, and confidence.
- No file over 100 MB enters Git.
- No secret or licence manifest enters Git.

### Cluster

- The container builds.
- The ADT sample loads in a shared world frame.
- The label audit tool produces a reproducible report.
- Feature extraction writes a cache.
- Bayes and EvidenceGRU smoke tests complete.
- One evaluation run writes a `RunManifest`.
- The cluster run produces a Pareto plot.

## 8. Final Deliverables

- Private GitHub repository named `wise-egomap`.
- README with a 100-word abstract.
- Architecture and map-lifecycle diagram.
- Technical report.
- Results table and error analysis.
- Limitations section.
- 90-second demo video.
- Interactive replay.
- Exact ADT access and download instructions.
- Exact cluster commands.
- Release tag.

## 9. Confirmed Answers

1. Target: complete code, cluster package, and synthetic demo first.
2. Real training starts only after the demo passes.
3. Scheduler: Slurm.
4. Compute: about 10 hours on H100 and H200 GPUs.
5. Compute nodes have internet access.
6. The user runs all cluster jobs.
7. GitHub account: `jagerknight10`.
8. Repository: private `wise-egomap`.
9. ADT licence: not approved.
10. ADT work is blocked until the licence is approved.

## 10. Defaults

1. Runtime: Apptainer.
2. Python: 3.11.
3. Package manager: `uv`.
4. Config: YAML.
5. Tests: pytest.
6. Lint and format: Ruff.
7. Keyframe rate: 2 Hz.
8. Seeds: `0, 1, 2`.
9. Backbones: frozen.
10. Viewer: static replay first, Rerun later.
11. Licence: MIT for code.

## 11. ADT Licence Rule

Do not download ADT data without a licence.

The synthetic benchmark is the first quantitative target. It has clear limits.
Do not present synthetic results as ADT results.

Read `docs/ADT_ACCESS.md` for the access steps and the fallback plan.

## 11. Start Condition

Implementation starts when the user confirms the defaults or answers the open questions.

