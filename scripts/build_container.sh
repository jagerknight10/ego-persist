#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

if [[ -z "${WISE_EGOMAP_IMAGE:-}" ]]; then
    printf 'ERROR: WISE_EGOMAP_IMAGE is not set.\n' >&2
    exit 1
fi
if [[ "$WISE_EGOMAP_IMAGE" != /* ]]; then
    printf 'ERROR: WISE_EGOMAP_IMAGE must be an absolute path.\n' >&2
    exit 1
fi
if [[ -e "$WISE_EGOMAP_IMAGE" ]]; then
    printf 'ERROR: Image already exists: %s\n' "$WISE_EGOMAP_IMAGE" >&2
    exit 1
fi
for path in \
    "${project_root}/pyproject.toml" \
    "${project_root}/uv.lock" \
    "${project_root}/environment/apptainer.def"; do
    if [[ ! -f "$path" ]]; then
        printf 'ERROR: Required build input is missing: %s\n' "$path" >&2
        exit 1
    fi
done
if ! command -v apptainer >/dev/null 2>&1; then
    printf 'ERROR: apptainer is not available.\n' >&2
    exit 1
fi

mkdir -p -- "$(dirname -- "$WISE_EGOMAP_IMAGE")"
cd "$project_root"
build_args=(--force)
if [[ "$(id -u)" -ne 0 ]]; then
    # Non-root build, as in CI. Cluster login nodes build as root.
    build_args+=(--fakeroot)
fi
apptainer build "${build_args[@]}" "$WISE_EGOMAP_IMAGE" environment/apptainer.def
