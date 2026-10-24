#!/usr/bin/env bash
#SBATCH --job-name=wise_egomap-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

prepare_job features 1
run_module_stage features WISE_EGOMAP_FEATURES_MODULE
