# Infrastructure Agent Brief

## Mission

Make cluster runs reproducible and safe.

## Owned Paths

- `environment`
- `jobs`
- `scripts`
- `.github/workflows`
- `docs/CLUSTER_RUNBOOK.md`

## Tasks

1. Pin the Python environment.
2. Build the Apptainer recipe.
3. Build Slurm templates.
4. Add run provenance.
5. Add CI.
6. Add large-file checks.
7. Add secret checks.
8. Add cluster helper scripts.
9. Add the ADT download wrapper.

## Rules

- All paths come from environment variables.
- No cluster secrets in Git.
- No data in Git.
- Use `set -euo pipefail`.
- Make scripts shellcheck-clean.
- Fail early when a path is missing.
- Keep raw data read-only.

## Checks

- shell syntax;
- Python compile;
- CI config parse;
- file-size gate;
- secret pattern gate.
