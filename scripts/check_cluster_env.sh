#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cluster_common.sh
source "${script_dir}/cluster_common.sh"

for name in \
    WISE_EGOMAP_PROJECT_ROOT \
    WISE_EGOMAP_DATA_ROOT \
    WISE_EGOMAP_RUN_ROOT \
    WISE_EGOMAP_IMAGE \
    WISE_EGOMAP_DATA_CONFIG \
    WISE_EGOMAP_LABEL_CONFIG \
    WISE_EGOMAP_EXPERIMENT_CONFIG \
    WISE_EGOMAP_MODEL_CONFIG \
    WISE_EGOMAP_SPLIT_FILE; do
    require_env "$name"
done

for name in \
    WISE_EGOMAP_PROJECT_ROOT \
    WISE_EGOMAP_DATA_ROOT \
    WISE_EGOMAP_RUN_ROOT \
    WISE_EGOMAP_IMAGE \
    WISE_EGOMAP_DATA_CONFIG \
    WISE_EGOMAP_LABEL_CONFIG \
    WISE_EGOMAP_EXPERIMENT_CONFIG \
    WISE_EGOMAP_MODEL_CONFIG \
    WISE_EGOMAP_SPLIT_FILE; do
    require_absolute_path "${!name}" "$name"
    require_clean_bind_path "${!name}" "$name"
done

require_directory "$WISE_EGOMAP_PROJECT_ROOT" "WISE_EGOMAP_PROJECT_ROOT"
require_directory "$WISE_EGOMAP_DATA_ROOT" "WISE_EGOMAP_DATA_ROOT"
require_writable_directory "$WISE_EGOMAP_RUN_ROOT" "WISE_EGOMAP_RUN_ROOT"
require_file "$WISE_EGOMAP_IMAGE" "WISE_EGOMAP_IMAGE"
require_file "$WISE_EGOMAP_DATA_CONFIG" "WISE_EGOMAP_DATA_CONFIG"
require_file "$WISE_EGOMAP_LABEL_CONFIG" "WISE_EGOMAP_LABEL_CONFIG"
require_file "$WISE_EGOMAP_EXPERIMENT_CONFIG" "WISE_EGOMAP_EXPERIMENT_CONFIG"
require_file "$WISE_EGOMAP_MODEL_CONFIG" "WISE_EGOMAP_MODEL_CONFIG"
require_file "$WISE_EGOMAP_SPLIT_FILE" "WISE_EGOMAP_SPLIT_FILE"
require_command apptainer

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
