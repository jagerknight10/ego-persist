# Environment

## Runtime

| Item | Value |
|---|---|
| Python | 3.11 |
| Package manager | `uv` 0.8.13 |
| Container | Apptainer |
| Base image | `python:3.11.14-slim-bookworm` |

The base image uses a tag. Before a release build, replace the tag with an
image digest. A digest gives an exact base image.

## Make the lock file

Do this step on a machine with network access. The cluster login node is
suitable. Do not use this laptop.

```bash
cd "$EGOPERSIST_PROJECT_ROOT"
bash scripts/lock_environment.sh
git add uv.lock
```

`scripts/lock_environment.sh` uses Python 3.11 and the `uv` version in
`environment/uv-version.txt`. The script fails when `uv` is absent.

`uv.lock` is not a secret. Commit it. Do not commit `.venv` or a package cache.

If a dependency range changes in `pyproject.toml`, make a new lock file. Do
not edit the lock file by hand.

## Build the container

Set these paths before the build:

```bash
export EGOPERSIST_PROJECT_ROOT=<ABSOLUTE_REPO_PATH>
export EGOPERSIST_IMAGE=<ABSOLUTE_SIF_PATH>
```

Build on the cluster:

```bash
cd "$EGOPERSIST_PROJECT_ROOT"
apptainer build "$EGOPERSIST_IMAGE" environment/apptainer.def
```

The recipe copies `uv.lock` into the image. A missing lock file stops the
build.

## Test the container

```bash
apptainer test "$EGOPERSIST_IMAGE"
```

The test runs the repository test suite in the image.

## Refresh the lock file

1. Change `pyproject.toml` through the lead.
2. Run `bash scripts/lock_environment.sh`.
3. Review the lock diff.
4. Run the test suite.
5. Build a new image.

Do not put URLs, tokens, data paths, or download manifests in the image recipe.
