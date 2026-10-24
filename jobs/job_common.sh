#!/usr/bin/env bash
# Shared checks for Slurm stage templates.

prepare_job() {
    local stage="$1"
    local gpu_required="$2"
    local base_run_id
    local run_id

    require_env WISE_EGOMAP_PROJECT_ROOT
    require_absolute_path "$WISE_EGOMAP_PROJECT_ROOT" "WISE_EGOMAP_PROJECT_ROOT"
    require_directory "$WISE_EGOMAP_PROJECT_ROOT" "WISE_EGOMAP_PROJECT_ROOT"
    bash "${WISE_EGOMAP_PROJECT_ROOT}/scripts/check_cluster_env.sh" >/dev/null

    # One experiment = one base run id. Set WISE_EGOMAP_RUN_ID in the
    # cluster environment file. All stages except train share it.
    require_env WISE_EGOMAP_RUN_ID
    base_run_id="$WISE_EGOMAP_RUN_ID"
    if [[ ! "$base_run_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
        [[ "$base_run_id" == "." || "$base_run_id" == ".." ]]; then
        cluster_fail "WISE_EGOMAP_RUN_ID has unsafe characters."
    fi
    export WISE_EGOMAP_BASE_RUN_ID="$base_run_id"

    export WISE_EGOMAP_SEED_BASE="${WISE_EGOMAP_SEED_BASE:-0}"
    if [[ ! "$WISE_EGOMAP_SEED_BASE" =~ ^[0-9]+$ ]]; then
        cluster_fail "WISE_EGOMAP_SEED_BASE must be a non-negative integer."
    fi
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        export WISE_EGOMAP_SEED="$((WISE_EGOMAP_SEED_BASE + SLURM_ARRAY_TASK_ID))"
    else
        export WISE_EGOMAP_SEED="$WISE_EGOMAP_SEED_BASE"
    fi

    if [[ "$stage" == "train" ]]; then
        # Each seed owns its run directory; it reads features from the
        # base run via WISE_EGOMAP_BASE_RUN_ID.
        run_id="${base_run_id}_train_seed_${WISE_EGOMAP_SEED}"
    elif [[ "$stage" == "eval" ]]; then
        # Eval reads shared data from the base run and writes to its own
        # directory so model and Bayes variants never collide.
        if [[ -n "${WISE_EGOMAP_MODEL_RUN_ID:-}" ]]; then
            run_id="${base_run_id}_eval_model"
        else
            run_id="${base_run_id}_eval_bayes"
        fi
    else
        run_id="$base_run_id"
    fi
    export WISE_EGOMAP_RUN_ID="$run_id"

    export WISE_EGOMAP_USE_GPU="$gpu_required"

    if [[ "$gpu_required" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        cluster_fail "GPU is necessary, but CUDA_VISIBLE_DEVICES is empty."
    fi
}

run_module_stage() {
    local stage="$1"
    local module="$2"

    require_env "$module"
    local module_path="${!module}"
    if [[ ! "$module_path" =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]]; then
        cluster_fail "${module} is not a valid Python module: ${module_path}"
    fi
    local stage_script
    stage_script="exec python -m \"\$${module}\" --stage \"${stage}\""
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        bash "${WISE_EGOMAP_PROJECT_ROOT}/scripts/run_stage.sh" \
        "$stage" -- bash -c "$stage_script"
}
