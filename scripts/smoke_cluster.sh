# Cluster smoke test. Run on one compute node before submitting stage jobs.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cluster_common.sh
source "${script_dir}/cluster_common.sh"

bash "${script_dir}/check_cluster_env.sh"

image_sha256="$(sha256sum "$EGOPERSIST_IMAGE" | awk '{print $1}')"
printf 'Smoke image SHA-256: %s\n' "$image_sha256"

# The workspace bind is read-only, so pytest must not write a cache.
apptainer exec \
    --cleanenv --no-home \
    --bind "${EGOPERSIST_PROJECT_ROOT}:/workspace:ro" \
    --bind "${EGOPERSIST_DATA_ROOT}:/data:ro" \
    --bind "${EGOPERSIST_RUN_ROOT}:/runs" \
    --env "EGOPERSIST_PROJECT_ROOT=/workspace" \
    --env "EGOPERSIST_DATA_ROOT=/data" \
    --env "EGOPERSIST_RUN_ROOT=/runs" \
    --env "PYTHONPATH=/workspace/src" \
    --nv \
    "$EGOPERSIST_IMAGE" \
    bash -c 'cd /workspace \
        && python -c "import egopersist; print(egopersist.__version__)" \
        && python -m pytest tests -q -p no:cacheprovider'

printf 'Cluster smoke passed.\n'
