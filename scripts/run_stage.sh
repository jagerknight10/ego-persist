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
require_env WISE_EGOMAP_SEED
if [[ ! "$WISE_EGOMAP_SEED" =~ ^[0-9]+$ ]]; then
    cluster_fail "WISE_EGOMAP_SEED must be a non-negative integer."
fi

export WISE_EGOMAP_RUN_ID="${WISE_EGOMAP_RUN_ID:-$(default_run_id)}"
if [[ ! "$WISE_EGOMAP_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    [[ "$WISE_EGOMAP_RUN_ID" == "." || "$WISE_EGOMAP_RUN_ID" == ".." ]]; then
    cluster_fail "WISE_EGOMAP_RUN_ID has unsafe characters."
fi

case "$stage" in
    preprocess)
        stage_config="${WISE_EGOMAP_STAGE_CONFIG:-$WISE_EGOMAP_DATA_CONFIG}"
        ;;
    features | train | eval)
        stage_config="${WISE_EGOMAP_STAGE_CONFIG:-$WISE_EGOMAP_EXPERIMENT_CONFIG}"
        ;;
esac
require_file "$stage_config" "Stage config"

run_dir="$(make_run_directory "$WISE_EGOMAP_RUN_ROOT" "$WISE_EGOMAP_RUN_ID")"
install -m 640 -- "$stage_config" "${run_dir}/resolved_config.yaml"

if [[ "${WISE_EGOMAP_RUNTIME:-apptainer}" == "venv" ]]; then
    # The site has no container runtime. The Slurm job activates the shared
    # virtual environment before calling this script.
    exec "$@"
fi

config_path="/runs/${WISE_EGOMAP_RUN_ID}/resolved_config.yaml"
label_path="$(project_to_container "$WISE_EGOMAP_LABEL_CONFIG")"
split_path="$(data_to_container "$WISE_EGOMAP_SPLIT_FILE")"
image_sha256="$(sha256sum "$WISE_EGOMAP_IMAGE" | awk '{print $1}')"

apptainer_args=(
    exec
    --cleanenv
    --no-home
    --pwd /workspace
    --bind "${WISE_EGOMAP_PROJECT_ROOT}:/workspace:ro"
    --bind "${WISE_EGOMAP_DATA_ROOT}:/data:ro"
    --bind "${WISE_EGOMAP_RUN_ROOT}:/runs"
)

container_env=(
    "WISE_EGOMAP_PROJECT_ROOT=/workspace"
    "WISE_EGOMAP_DATA_ROOT=/data"
    "WISE_EGOMAP_RUN_ROOT=/runs"
    "WISE_EGOMAP_RUN_ID=${WISE_EGOMAP_RUN_ID}"
    "WISE_EGOMAP_SEED=${WISE_EGOMAP_SEED}"
    "WISE_EGOMAP_DATA_CONFIG=$(project_to_container "$WISE_EGOMAP_DATA_CONFIG")"
    "WISE_EGOMAP_LABEL_CONFIG=${label_path}"
    "WISE_EGOMAP_EXPERIMENT_CONFIG=$(project_to_container "$WISE_EGOMAP_EXPERIMENT_CONFIG")"
    "WISE_EGOMAP_MODEL_CONFIG=$(project_to_container "$WISE_EGOMAP_MODEL_CONFIG")"
    "WISE_EGOMAP_SPLIT_FILE=${split_path}"
    "PYTHONPATH=/workspace/src"
)
for name in \
    SLURM_JOB_ID \
    SLURM_ARRAY_JOB_ID \
    SLURM_ARRAY_TASK_ID \
    WISE_EGOMAP_BASE_RUN_ID \
    WISE_EGOMAP_MODEL_RUN_ID \
    WISE_EGOMAP_PREPROCESS_MODULE \
    WISE_EGOMAP_FEATURES_MODULE \
    WISE_EGOMAP_TRAIN_MODULE \
    WISE_EGOMAP_EVAL_MODULE; do
    if [[ -n "${!name:-}" ]]; then
        container_env+=("${name}=${!name}")
    fi
done
for item in "${container_env[@]}"; do
    apptainer_args+=(--env "$item")
done

if [[ "${WISE_EGOMAP_USE_GPU:-0}" == "1" ]]; then
    apptainer_args+=(--nv)
fi

apptainer "${apptainer_args[@]}" "$WISE_EGOMAP_IMAGE" \
    python /workspace/scripts/provenance.py \
    --project-root /workspace \
    --run-id "$WISE_EGOMAP_RUN_ID" \
    --stage "$stage" \
    --config "$config_path" \
    --data-split "$split_path" \
    --label-config "$label_path" \
    --seed "$WISE_EGOMAP_SEED" \
    --container-image "$WISE_EGOMAP_IMAGE" \
    --container-sha256 "$image_sha256" \
    --output "/runs/${WISE_EGOMAP_RUN_ID}/provenance.json" \
    --command "$@"

apptainer "${apptainer_args[@]}" "$WISE_EGOMAP_IMAGE" "$@"
