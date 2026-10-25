#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cluster_common.sh
source "${script_dir}/cluster_common.sh"

for name in \
    WISE_EGOMAP_PROJECT_ROOT \
    WISE_EGOMAP_DATA_ROOT \
    WISE_EGOMAP_RUN_ROOT \
    WISE_EGOMAP_DATA_CONFIG \
    WISE_EGOMAP_LABEL_CONFIG \
    WISE_EGOMAP_EXPERIMENT_CONFIG \
    WISE_EGOMAP_MODEL_CONFIG \
    WISE_EGOMAP_SPLIT_FILE; do
    require_env "$name"
done
runtime="${WISE_EGOMAP_RUNTIME:-apptainer}"
case "$runtime" in
    apptainer)
        require_env WISE_EGOMAP_IMAGE
        require_file "$WISE_EGOMAP_IMAGE" "WISE_EGOMAP_IMAGE"
        require_command apptainer
        ;;
    venv)
        require_env WISE_EGOMAP_VENV
        require_directory "$WISE_EGOMAP_VENV" "WISE_EGOMAP_VENV"
        ;;
    *)
        cluster_fail "WISE_EGOMAP_RUNTIME must be apptainer or venv."
        ;;
esac

for name in \
    WISE_EGOMAP_PROJECT_ROOT \
    WISE_EGOMAP_DATA_ROOT \
    WISE_EGOMAP_RUN_ROOT \
    WISE_EGOMAP_DATA_CONFIG \
    WISE_EGOMAP_LABEL_CONFIG \
    WISE_EGOMAP_EXPERIMENT_CONFIG \
    WISE_EGOMAP_MODEL_CONFIG \
    WISE_EGOMAP_SPLIT_FILE; do
    require_absolute_path "${!name}" "$name"
    require_clean_bind_path "${!name}" "$name"
done
if [[ "$runtime" == "apptainer" ]]; then
    require_absolute_path "$WISE_EGOMAP_IMAGE" "WISE_EGOMAP_IMAGE"
fi

require_directory "$WISE_EGOMAP_PROJECT_ROOT" "WISE_EGOMAP_PROJECT_ROOT"
require_directory "$WISE_EGOMAP_DATA_ROOT" "WISE_EGOMAP_DATA_ROOT"
require_writable_directory "$WISE_EGOMAP_RUN_ROOT" "WISE_EGOMAP_RUN_ROOT"
require_file "$WISE_EGOMAP_DATA_CONFIG" "WISE_EGOMAP_DATA_CONFIG"
require_file "$WISE_EGOMAP_LABEL_CONFIG" "WISE_EGOMAP_LABEL_CONFIG"
require_file "$WISE_EGOMAP_EXPERIMENT_CONFIG" "WISE_EGOMAP_EXPERIMENT_CONFIG"
require_file "$WISE_EGOMAP_MODEL_CONFIG" "WISE_EGOMAP_MODEL_CONFIG"
require_file "$WISE_EGOMAP_SPLIT_FILE" "WISE_EGOMAP_SPLIT_FILE"

for path in \
    "$WISE_EGOMAP_DATA_CONFIG" \
    "$WISE_EGOMAP_LABEL_CONFIG" \
    "$WISE_EGOMAP_EXPERIMENT_CONFIG" \
    "$WISE_EGOMAP_MODEL_CONFIG"; do
    project_to_container "$path" >/dev/null
done

project_to_container "$WISE_EGOMAP_PROJECT_ROOT/pyproject.toml" >/dev/null
data_to_container "$WISE_EGOMAP_SPLIT_FILE" >/dev/null
require_outside_path "$WISE_EGOMAP_RUN_ROOT" "$WISE_EGOMAP_DATA_ROOT" "WISE_EGOMAP_RUN_ROOT"
require_outside_path "$WISE_EGOMAP_RUN_ROOT" "$WISE_EGOMAP_PROJECT_ROOT" "WISE_EGOMAP_RUN_ROOT"

printf 'Cluster environment is valid.\n'
