# Agent Coordination

The build uses one lead and six workstreams.

## Rules

1. The lead freezes shared contracts first.
2. Each agent uses its own Git worktree and branch.
3. Each agent changes only its owned paths.
4. Each agent adds tests for its own code.
5. Each agent reports changed files, test results, and blockers.
6. The lead merges in the order in `docs/PLAN.md`.
7. No agent downloads datasets or large models.
8. No agent commits secrets, manifests, data, or checkpoints.

## Branches

| Agent | Branch |
|---|---|
| Lead | `main` |
| Data | `feat/data` |
| Geometry | `feat/geometry` |
| Evidence | `feat/evidence` |
| Evaluation | `feat/evaluation` |
| Demo | `feat/demo` |
| Infra | `feat/infra` |

## Definition of Done

An agent is done when:

- its owned code is complete;
- tests pass on CPU;
- public functions have type hints;
- errors use explicit exceptions;
- output schemas are stable;
- no large file enters Git;
- the agent report is complete.
