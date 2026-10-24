# EgoPersist Task Board

The plan is accepted. Work starts with the synthetic path.

ADT tasks remain blocked until the licence is approved.

## Phase 0: Contract Freeze

| ID | Owner | Task | Depends on | Done when |
|---|---|---|---|---|
| P0-1 | Lead | Confirm scope and defaults | User | Answers recorded |
| P0-2 | Lead | Create repo scaffold | P0-1 | `src` layout and CI exist |
| P0-3 | Lead | Define shared contracts | P0-2 | Contract tests pass |
| P0-4 | Lead | Create agent worktrees | P0-3 | One branch per agent |

### Synthetic-First Gate

| ID | Owner | Task | Depends on | Done when |
|---|---|---|---|---|
| S1 | Demo | Build three synthetic scenarios | P0-3 | All cases replay |
| S2 | Geometry | Run state machine on scenarios | S1 | Actions are correct |
| S3 | Evaluation | Score the scenarios | S2 | Report is stable |
| S4 | Lead | Review the demo | S3 | Demo accepted |
| S5 | Lead | Decide on real training | S4 | Decision recorded |

## Phase 1: Data and Geometry

| ID | Owner | Task | Depends on | Done when |
|---|---|---|---|---|
| D1 | Data | Build tiny synthetic fixture | P0-3 | Fixture under 10 MB |
| D2 | Data | Implement ADT adapter | D1 | Reader test passes |
| D3 | Data | Implement label rules | D2 | Rule tests pass |
| D4 | Data | Build audit tool | D2 | Audit report generated |
| G1 | Geometry | Implement depth back-projection | P0-3 | Round-trip test passes |
| G2 | Geometry | Implement track association | G1 | Hungarian gate test passes |
| G3 | Geometry | Implement map state machine | G2 | Transition tests pass |
| G4 | Geometry | Implement commit, defer, reject, revert | G3 | Decision tests pass |

## Phase 2: Evidence and Evaluation

| ID | Owner | Task | Depends on | Done when |
|---|---|---|---|---|
| E1 | Evidence | Build evidence vector | P0-3, G2 | Schema test passes |
| E2 | Evidence | Implement Bayes baseline | E1 | Fixture scores match |
| E3 | Evidence | Implement EvidenceGRU | E1 | Forward pass works |
| E4 | Evidence | Add calibration and loss | E3 | Train smoke test passes |
| V1 | Evaluation | Implement metrics | P0-3 | Metric tests pass |
| V2 | Evaluation | Implement event evaluator | V1 | Fixture report matches |
| V3 | Evaluation | Build Pareto plot | V2 | Plot generated |
| V4 | Evaluation | Build error taxonomy | V2 | False commits classified |

## Phase 3: Baselines and Ablations

| ID | Owner | Task | Depends on | Done when |
|---|---|---|---|---|
| B1 | Evidence | Never-update baseline | G3 | Runs on fixture |
| B2 | Evidence | Instant geometric baseline | G3 | Runs on fixture |
| B3 | Evidence | Instant semantic baseline | G3 | Runs on fixture |
| B4 | Evidence | Majority-vote baseline | E1 | Runs on fixture |
| B5 | Evidence | Bayes baseline | E2 | Runs on fixture |
| B6 | Evidence | EvidenceGRU baseline | E4 | Runs on fixture |
| A1 | Evidence | Required ablation configs | B6 | Six configs exist |

## Phase 4: Demo

| ID | Owner | Task | Depends on | Done when |
|---|---|---|---|---|
| M1 | Demo | Build replay data schema | P0-3 | Schema test passes |
| M2 | Demo | Build scenario replay | M1, G4 | Three cases run |
| M3 | Demo | Build 3D viewer | M2 | Map and timeline show |
| M4 | Demo | Build query flow | M3 | Mug query returns state |
| M5 | Demo | Build side-by-side comparison | M3 | Instant vs EgoPersist |
| M6 | Demo | Build video scaffold | M5 | 90-second render path |

## Phase 5: Infrastructure and Release

| ID | Owner | Task | Depends on | Done when |
|---|---|---|---|---|
| I1 | Infra | Pin Python environment | P0-2 | Lockfile exists |
| I2 | Infra | Build Apptainer recipe | I1 | Build instructions tested |
| I3 | Infra | Build Slurm job templates | I2 | Shellcheck passes |
| I4 | Infra | Add run manifest and provenance | P0-3 | JSON output exists |
| I5 | Infra | Add CI checks | P0-2 | CI passes |
| I6 | Infra | Add large-file and secret checks | P0-2 | Pre-commit passes |
| R1 | Lead | Write README and abstract | All | Review complete |
| R2 | Lead | Write technical report | V3 | Review complete |
| R3 | Lead | Write cluster runbook | I3 | Commands tested |
| R4 | Lead | Write limitations | R2 | Review complete |
| R5 | Lead | Create release candidate | All | CI clean |

## Cluster Gate

| ID | Owner | Task | Depends on | Done when |
|---|---|---|---|---|
| C1 | User | Approve ADT licence | None | Licence active |
| C2 | User | Provide cluster paths | None | Values recorded |
| C3 | User | Place ADT manifest | C1 | Manifest on cluster |
| C4 | User | Run sample download | C3 | Sample files exist |
| C5 | User | Run smoke test | C4, R3 | Tests pass |
| C6 | User | Run label pipeline | C5 | Labels and audit exist |
| C7 | User | Run feature extraction | C6 | Feature cache exists |
| C8 | User | Run training | C7 | Three seeds complete |
| C9 | User | Run evaluation | C8 | Pareto plot exists |
