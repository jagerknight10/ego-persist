#!/usr/bin/env bash
#SBATCH --job-name=wise_egomap-demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

source "${WISE_EGOMAP_PROJECT_ROOT}/scripts/load_cluster_env.sh"
source "${WISE_EGOMAP_VENV}/bin/activate"

python "${WISE_EGOMAP_PROJECT_ROOT}/scripts/render_adt_demo.py" \
  --data-root "${WISE_EGOMAP_DATA_ROOT}" \
  --run-root "${WISE_EGOMAP_RUN_ROOT}" \
  --eval-run-id "${WISE_EGOMAP_DEMO_EVAL_RUN_ID}" \
  --sequence "${WISE_EGOMAP_DEMO_SEQUENCE}" \
  --output "${WISE_EGOMAP_DEMO_OUTPUT}" \
  --duration-s "${WISE_EGOMAP_DEMO_DURATION_S:-16}" \
  --fps "${WISE_EGOMAP_DEMO_FPS:-6}"
