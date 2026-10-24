#!/usr/bin/env bash
# Shared helpers for cluster entry points. Callers enable strict mode.

cluster_fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        cluster_fail "Required environment variable is unset: ${name}"
    fi
}

require_command() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        cluster_fail "Required command is not available: ${name}"
    fi
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "$path" ]]; then
        cluster_fail "${label} is not a file: ${path}"
    fi
    if [[ ! -r "$path" ]]; then
        cluster_fail "${label} is not readable: ${path}"
    fi
}

require_directory() {
    local path="$1"
    local label="$2"
    if [[ ! -d "$path" ]]; then
        cluster_fail "${label} is not a directory: ${path}"
    fi
    if [[ ! -r "$path" || ! -x "$path" ]]; then
        cluster_fail "${label} is not readable: ${path}"
    fi
}

require_writable_directory() {
    local path="$1"
    local label="$2"
    require_directory "$path" "$label"
    if [[ ! -w "$path" ]]; then
        cluster_fail "${label} is not writable: ${path}"
    fi
}

canonical_path() {
    local path="$1"
    local label="$2"
    local result
    if result="$(realpath -e -- "$path" 2>/dev/null)"; then
        printf '%s' "$result"
        return
    fi
    if result="$(realpath "$path" 2>/dev/null)" && [[ -e "$result" ]]; then
        printf '%s' "$result"
        return
    fi
    cluster_fail "Cannot resolve ${label}: ${path}"
}

require_absolute_path() {
    local path="$1"
    local label="$2"
    if [[ "$path" != /* ]]; then
        cluster_fail "${label} must be an absolute path: ${path}"
    fi
}

require_clean_bind_path() {
    local path="$1"
    local label="$2"
    if [[ "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
        cluster_fail "${label} cannot contain a line break: ${path}"
    fi
    case "$path" in
        *,* | *:*)
            cluster_fail "${label} cannot contain a comma or colon: ${path}"
            ;;
    esac
}

require_outside_path() {
    local child="$1"
    local parent="$2"
    local label="$3"
    local child_real
    local parent_real
    child_real="$(canonical_path "$child" "$label")"
    parent_real="$(canonical_path "$parent" "$label parent")"
    case "${child_real}/" in
        "${parent_real}/"*)
            cluster_fail "${label} must be outside ${parent}: ${child}"
            ;;
    esac
}

default_run_id() {
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
        printf '%s_%s' "$SLURM_ARRAY_JOB_ID" "${SLURM_ARRAY_TASK_ID:-0}"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
        printf '%s' "$SLURM_JOB_ID"
    else
        date -u '+%Y%m%dT%H%M%SZ'
    fi
}

make_run_directory() {
    local run_root="$1"
    local run_id="$2"
    local run_dir="${run_root}/${run_id}"
    if [[ -e "$run_dir" && ! -d "$run_dir" ]]; then
        cluster_fail "Run path is not a directory: ${run_dir}"
    fi
    mkdir -p -- "$run_dir"
    chmod 750 "$run_dir"
    printf '%s' "$run_dir"
}

project_to_container() {
    local path="$1"
    local root_real
    local path_real
    root_real="$(canonical_path "$WISE_EGOMAP_PROJECT_ROOT" "WISE_EGOMAP_PROJECT_ROOT")"
    path_real="$(canonical_path "$path" "project path")"
    case "$path_real" in
        "${root_real}/"*)
            printf '/workspace/%s' "${path_real#"${root_real}/"}"
            ;;
        *)
            cluster_fail "Path is not under WISE_EGOMAP_PROJECT_ROOT: ${path}"
            ;;
    esac
}

data_to_container() {
    local path="$1"
    local root_real
    local path_real
    root_real="$(canonical_path "$WISE_EGOMAP_DATA_ROOT" "WISE_EGOMAP_DATA_ROOT")"
    path_real="$(canonical_path "$path" "data path")"
    case "$path_real" in
        "${root_real}/"*)
            printf '/data/%s' "${path_real#"${root_real}/"}"
            ;;
        *)
            cluster_fail "Path is not under WISE_EGOMAP_DATA_ROOT: ${path}"
            ;;
    esac
}

to_container_path() {
    local path="$1"
    local project_real
    local data_real
    local run_real
    local path_real
    project_real="$(canonical_path "$WISE_EGOMAP_PROJECT_ROOT" "WISE_EGOMAP_PROJECT_ROOT")"
    data_real="$(canonical_path "$WISE_EGOMAP_DATA_ROOT" "WISE_EGOMAP_DATA_ROOT")"
    run_real="$(canonical_path "$WISE_EGOMAP_RUN_ROOT" "WISE_EGOMAP_RUN_ROOT")"
    path_real="$(canonical_path "$path" "bind path")"
    case "$path_real" in
        "${project_real}/"*)
            printf '/workspace/%s' "${path_real#"${project_real}/"}"
            ;;
        "${data_real}/"*)
            printf '/data/%s' "${path_real#"${data_real}/"}"
            ;;
        "${run_real}/"*)
            printf '/runs/%s' "${path_real#"${run_real}/"}"
            ;;
        *)
            cluster_fail "Path is not under a bind root: ${path}"
            ;;
    esac
}

project_file_to_container() {
    local path="$1"
    local label="$2"
    require_file "$path" "$label"
    project_to_container "$path"
}

data_file_to_container() {
    local path="$1"
    local label="$2"
    require_file "$path" "$label"
    data_to_container "$path"
}

run_path_to_container() {
    local path="$1"
    require_writable_directory "$WISE_EGOMAP_RUN_ROOT" "WISE_EGOMAP_RUN_ROOT"
    to_container_path "$path"
}
