# WISE-EgoMap Agent Rules

## Language

Use ASD-STE100 Simplified Technical English.

Keep answers short unless the user asks for an explanation.

## Scope

1. Build code, tests, configs, docs, and a synthetic demo on this machine.
2. Run all dataset downloads, preprocessing at scale, training, sweeps, and
   reported experiments on the university cluster.
3. Do not download any dataset or model file over 100 MB on this machine.
4. Do not commit raw data, checkpoints, download manifests, or secrets.
5. Do not claim ADT results until the ADT licence is approved.

## Engineering Rules

1. Use Python 3.11 or 3.12.
2. Use typed public functions.
3. Add a test for each new behaviour.
4. Keep modules separated by the contracts in `src/wise_egomap/contracts`.
5. Use SI units and put the unit in the field name.
6. Never use future frames in causal decisions.
7. Never use ground-truth object pose as a model input.
8. Keep fixtures small and deterministic.
9. Run `pytest` before you report completion.
10. Report changed files, test results, and blockers.

## Agent Boundaries

Each agent changes only its owned paths. See `docs/agents/`.

The lead owns shared interfaces. Agents must not change shared contracts
without lead approval.
