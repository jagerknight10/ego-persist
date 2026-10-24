#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cluster_common.sh
source "${script_dir}/cluster_common.sh"

usage() {
    cat >&2 <<'EOF'
Usage: run_stage.sh STAGE [--] COMMAND [ARG ...]

STAGE is one of: preprocess, features, train, eval.
All site paths come from environment variables. See
environment/cluster.env.example.
EOF
    exit 2
}

[[ $# -ge 2 ]] || usage
stage="$1"
shift
if [[ "${1:-}" == "--" ]]; then
    shift
fi
[[ $# -ge 1 ]] || usage

case "$stage" in
    preprocess | features | train | eval) ;;
    *)
        cluster_fail "Unknown stage: ${stage}"
        ;;
esac

bash "${script_dir}/check_cluster_env.sh" >/dev/null
require_command sha256sum
require_env EGOPERSIST_SEED
if [[ ! "$EGOPERSIST_SEED" =~ ^[0-9]+$ ]]; then
    cluster_fail "EGOPERSIST_SEED must be a non-negative integer."
fi

export EGOPERSIST_RUN_ID="${EGOPERSIST_RUN_ID:-$(default_run_id)}"
if [[ ! "$EGOPERSIST_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    [[ "$EGOPERSIST_RUN_ID" == "." || "$EGOPERSIST_RUN_ID" == ".." ]]; then
    cluster_fail "EGOPERSIST_RUN_ID has unsafe characters."
fi

case "$stage" in
    preprocess)
        stage_config="${EGOPERSIST_STAGE_CONFIG:-$EGOPERSIST_DATA_CONFIG}"
        ;;
    features | train | eval)
        stage_config="${EGOPERSIST_STAGE_CONFIG:-$EGOPERSIST_EXPERIMENT_CONFIG}"
        ;;
esac
require_file "$stage_config" "Stage config"

run_dir="$(make_run_directory "$EGOPERSIST_RUN_ROOT" "$EGOPERSIST_RUN_ID")"
install -m 640 -- "$stage_config" "${run_dir}/resolved_config.yaml"

config_path="/runs/${EGOPERSIST_RUN_ID}/resolved_config.yaml"
label_path="$(project_to_container "$EGOPERSIST_LABEL_CONFIG")"
split_path="$(data_to_container "$EGOPERSIST_SPLIT_FILE")"
image_sha256="$(sha256sum "$EGOPERSIST_IMAGE" | awk '{print $1}')"

apptainer_args=(
    exec
    --cleanenv
    --no-home
    --pwd /workspace
    --bind "${EGOPERSIST_PROJECT_ROOT}:/workspace:ro"
    --bind "${EGOPERSIST_DATA_ROOT}:/data:ro"
    --bind "${EGOPERSIST_RUN_ROOT}:/runs"
)

container_env=(
    "EGOPERSIST_PROJECT_ROOT=/workspace"
    "EGOPERSIST_DATA_ROOT=/data"
    "EGOPERSIST_RUN_ROOT=/runs"
    "EGOPERSIST_RUN_ID=${EGOPERSIST_RUN_ID}"
    "EGOPERSIST_SEED=${EGOPERSIST_SEED}"
    "EGOPERSIST_DATA_CONFIG=$(project_to_container "$EGOPERSIST_DATA_CONFIG")"
    "EGOPERSIST_LABEL_CONFIG=${label_path}"
    "EGOPERSIST_EXPERIMENT_CONFIG=$(project_to_container "$EGOPERSIST_EXPERIMENT_CONFIG")"
    "EGOPERSIST_MODEL_CONFIG=$(project_to_container "$EGOPERSIST_MODEL_CONFIG")"
    "EGOPERSIST_SPLIT_FILE=${split_path}"
    "PYTHONPATH=/workspace/src"
)
for name in \
    SLURM_JOB_ID \
    SLURM_ARRAY_JOB_ID \
    SLURM_ARRAY_TASK_ID \
    EGOPERSIST_BASE_RUN_ID \
    EGOPERSIST_MODEL_RUN_ID \
    EGOPERSIST_PREPROCESS_MODULE \
    EGOPERSIST_FEATURES_MODULE \
    EGOPERSIST_TRAIN_MODULE \
    EGOPERSIST_EVAL_MODULE; do
    if [[ -n "${!name:-}" ]]; then
        container_env+=("${name}=${!name}")
    fi
done
for item in "${container_env[@]}"; do
    apptainer_args+=(--env "$item")
done

if [[ "${EGOPERSIST_USE_GPU:-0}" == "1" ]]; then
    apptainer_args+=(--nv)
fi

apptainer "${apptainer_args[@]}" "$EGOPERSIST_IMAGE" \
    python /workspace/scripts/provenance.py \
    --project-root /workspace \
    --run-id "$EGOPERSIST_RUN_ID" \
    --stage "$stage" \
    --config "$config_path" \
    --data-split "$split_path" \
    --label-config "$label_path" \
    --seed "$EGOPERSIST_SEED" \
    --container-image "$EGOPERSIST_IMAGE" \
    --container-sha256 "$image_sha256" \
    --output "/runs/${EGOPERSIST_RUN_ID}/provenance.json" \
    --command "$@"

apptainer "${apptainer_args[@]}" "$EGOPERSIST_IMAGE" "$@"
