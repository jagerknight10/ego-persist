#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cluster_common.sh
source "${script_dir}/cluster_common.sh"

usage() {
    cat >&2 <<'EOF'
Usage: submit_stage.sh STAGE [--array=TASKS]

STAGE is one of: preprocess, features, train, eval.
Source the cluster environment file before use.
EOF
    exit 2
}

[[ $# -ge 1 ]] || usage
stage="$1"
shift
case "$stage" in
    preprocess | features | train | eval) ;;
    *)
        cluster_fail "Unknown stage: ${stage}"
        ;;
esac

array_tasks=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --array=*)
            array_tasks="${1#*=}"
            shift
            ;;
        *)
            usage
            ;;
    esac
done
if [[ -n "$array_tasks" && ! "$array_tasks" =~ ^[0-9]+(-[0-9]+)?(:[0-9]+)?$ ]]; then
    cluster_fail "Array task range is not valid: ${array_tasks}"
fi

bash "${script_dir}/check_cluster_env.sh" >/dev/null
require_env WISE_EGOMAP_SLURM_PARTITION
require_command sbatch

template="${WISE_EGOMAP_PROJECT_ROOT}/jobs/${stage}.sbatch"
require_file "$template" "Slurm template"
log_root="${WISE_EGOMAP_RUN_ROOT}/slurm"
if [[ ! -d "$log_root" ]]; then
    mkdir -- "$log_root"
    chmod 750 "$log_root"
fi

sbatch_args=(
    "--partition=${WISE_EGOMAP_SLURM_PARTITION}"
    "--chdir=${WISE_EGOMAP_PROJECT_ROOT}"
    "--output=${log_root}/${stage}_%j.out"
    "--error=${log_root}/${stage}_%j.err"
)
if [[ -n "${WISE_EGOMAP_SLURM_ACCOUNT:-}" ]]; then
    sbatch_args+=("--account=${WISE_EGOMAP_SLURM_ACCOUNT}")
fi
if [[ -n "$array_tasks" ]]; then
    sbatch_args+=("--array=${array_tasks}")
fi

sbatch "${sbatch_args[@]}" "$template"
