#!/usr/bin/env bash
# Cluster smoke test. Run on one compute node before submitting stage jobs.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cluster_common.sh
source "${script_dir}/cluster_common.sh"

bash "${script_dir}/check_cluster_env.sh"

image_sha256="$(sha256sum "$WISE_EGOMAP_IMAGE" | awk '{print $1}')"
printf 'Smoke image SHA-256: %s\n' "$image_sha256"

# The workspace bind is read-only, so pytest must not write a cache.
apptainer exec \
    --cleanenv --no-home \
    --bind "${WISE_EGOMAP_PROJECT_ROOT}:/workspace:ro" \
    --bind "${WISE_EGOMAP_DATA_ROOT}:/data:ro" \
    --bind "${WISE_EGOMAP_RUN_ROOT}:/runs" \
    --env "WISE_EGOMAP_PROJECT_ROOT=/workspace" \
    --env "WISE_EGOMAP_DATA_ROOT=/data" \
    --env "WISE_EGOMAP_RUN_ROOT=/runs" \
    --env "PYTHONPATH=/workspace/src" \
    --nv \
    "$WISE_EGOMAP_IMAGE" \
    bash -c 'cd /workspace \
        && python -c "import wise_egomap; print(wise_egomap.__version__)" \
        && python -m pytest tests -q -p no:cacheprovider'

printf 'Cluster smoke passed.\n'
