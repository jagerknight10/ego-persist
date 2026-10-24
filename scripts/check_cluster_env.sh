#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cluster_common.sh
source "${script_dir}/cluster_common.sh"

for name in \
    EGOPERSIST_PROJECT_ROOT \
    EGOPERSIST_DATA_ROOT \
    EGOPERSIST_RUN_ROOT \
    EGOPERSIST_IMAGE \
    EGOPERSIST_DATA_CONFIG \
    EGOPERSIST_LABEL_CONFIG \
    EGOPERSIST_EXPERIMENT_CONFIG \
    EGOPERSIST_MODEL_CONFIG \
    EGOPERSIST_SPLIT_FILE; do
    require_env "$name"
done

for name in \
    EGOPERSIST_PROJECT_ROOT \
    EGOPERSIST_DATA_ROOT \
    EGOPERSIST_RUN_ROOT \
    EGOPERSIST_IMAGE \
    EGOPERSIST_DATA_CONFIG \
    EGOPERSIST_LABEL_CONFIG \
    EGOPERSIST_EXPERIMENT_CONFIG \
    EGOPERSIST_MODEL_CONFIG \
    EGOPERSIST_SPLIT_FILE; do
    require_absolute_path "${!name}" "$name"
    require_clean_bind_path "${!name}" "$name"
done

require_directory "$EGOPERSIST_PROJECT_ROOT" "EGOPERSIST_PROJECT_ROOT"
require_directory "$EGOPERSIST_DATA_ROOT" "EGOPERSIST_DATA_ROOT"
require_writable_directory "$EGOPERSIST_RUN_ROOT" "EGOPERSIST_RUN_ROOT"
require_file "$EGOPERSIST_IMAGE" "EGOPERSIST_IMAGE"
require_file "$EGOPERSIST_DATA_CONFIG" "EGOPERSIST_DATA_CONFIG"
require_file "$EGOPERSIST_LABEL_CONFIG" "EGOPERSIST_LABEL_CONFIG"
require_file "$EGOPERSIST_EXPERIMENT_CONFIG" "EGOPERSIST_EXPERIMENT_CONFIG"
require_file "$EGOPERSIST_MODEL_CONFIG" "EGOPERSIST_MODEL_CONFIG"
require_file "$EGOPERSIST_SPLIT_FILE" "EGOPERSIST_SPLIT_FILE"
require_command apptainer

for path in \
    "$EGOPERSIST_DATA_CONFIG" \
    "$EGOPERSIST_LABEL_CONFIG" \
    "$EGOPERSIST_EXPERIMENT_CONFIG" \
    "$EGOPERSIST_MODEL_CONFIG"; do
    project_to_container "$path" >/dev/null
done

project_to_container "$EGOPERSIST_PROJECT_ROOT/pyproject.toml" >/dev/null
data_to_container "$EGOPERSIST_SPLIT_FILE" >/dev/null
require_outside_path "$EGOPERSIST_RUN_ROOT" "$EGOPERSIST_DATA_ROOT" "EGOPERSIST_RUN_ROOT"
require_outside_path "$EGOPERSIST_RUN_ROOT" "$EGOPERSIST_PROJECT_ROOT" "EGOPERSIST_RUN_ROOT"

printf 'Cluster environment is valid.\n'
