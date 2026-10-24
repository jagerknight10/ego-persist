#!/usr/bin/env bash
#SBATCH --job-name=wise_egomap-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        printf 'ERROR: Required environment variable is unset: %s\n' "$name" >&2
        exit 1
    fi
}
require_env WISE_EGOMAP_PROJECT_ROOT
# shellcheck source=scripts/cluster_common.sh
source "${WISE_EGOMAP_PROJECT_ROOT}/scripts/cluster_common.sh"
# shellcheck source=jobs/job_common.sh
source "${WISE_EGOMAP_PROJECT_ROOT}/jobs/job_common.sh"

# The train stage also fits temperature on val and writes model_probs/.
export WISE_EGOMAP_MODEL_STAGE=1
prepare_job train 1
run_module_stage train WISE_EGOMAP_TRAIN_MODULE
