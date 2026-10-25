#!/usr/bin/env bash
# Submit the ordered WISE-EgoMap Slurm stages.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cluster_common.sh
source "${script_dir}/cluster_common.sh"

require_env WISE_EGOMAP_PROJECT_ROOT
require_env WISE_EGOMAP_SLURM_PARTITION
require_env WISE_EGOMAP_RUN_ROOT
require_env WISE_EGOMAP_RUN_ID
require_command sbatch
for stage in preprocess features train eval; do
    require_file "${WISE_EGOMAP_PROJECT_ROOT}/jobs/${stage}.sbatch" "${stage} job"
done
bash "${script_dir}/check_cluster_env.sh" >/dev/null

log_root="${WISE_EGOMAP_RUN_ROOT}/slurm"
mkdir -p -- "$log_root"
chmod 750 "$log_root"

common_args=(
    "--partition=${WISE_EGOMAP_SLURM_PARTITION}"
    "--chdir=${WISE_EGOMAP_PROJECT_ROOT}"
)
if [[ -n "${WISE_EGOMAP_SLURM_ACCOUNT:-}" ]]; then
    common_args+=("--account=${WISE_EGOMAP_SLURM_ACCOUNT}")
fi

submit() {
    local stage="$1"
    local dependency="$2"
    shift 2
    local args=(
        "${common_args[@]}"
        "--output=${log_root}/${stage}_%j.out"
        "--error=${log_root}/${stage}_%j.err"
    )
    if [[ -n "$dependency" ]]; then
        args+=("--dependency=afterok:${dependency}")
    fi
    sbatch --parsable "${args[@]}" "$@" "${WISE_EGOMAP_PROJECT_ROOT}/jobs/${stage}.sbatch"
}

preprocess_id="$(submit preprocess "")"
features_id="$(submit features "${preprocess_id}")"
train_id="$(submit train "${features_id}" --array=0-2)"
bayes_eval_id="$(submit eval "${features_id}")"
model_eval_id="$(submit eval "${train_id}" \
    "--export=ALL,WISE_EGOMAP_MODEL_RUN_ID=${WISE_EGOMAP_RUN_ID}_train_seed_0")"

printf 'preprocess=%s\nfeatures=%s\ntrain=%s\nbayes_eval=%s\nmodel_eval=%s\n' \
    "$preprocess_id" "$features_id" "$train_id" "$bayes_eval_id" "$model_eval_id"
